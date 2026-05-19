from ...utils import *
from ...core import *

# =========================================================
# Worker Functions (Improvement 1, 2, 3: 완전 분리형 아키텍처)
# =========================================================
def _single_tune_run(args, queue):
    worker_logger = setup_logger()
    import traceback, pandas as pd
    from pathlib import Path
    from ultralytics import YOLO
    
    try:
        model = YOLO(args["model_name"])
        if args.get("webhook_url") and args.get("noti_flags", {}).get("epoch", False):
            interval = args.get("noti_flags", {}).get("epoch_interval", 100)
            model.add_callback("on_train_epoch_end", create_heartbeat_callback(args["webhook_url"], args["adaptive_epochs"], interval))
            
        if args.get("stop_event"):
            for hook, cb in create_stop_callback(args["stop_event"]).items():
                model.add_callback(hook, cb)

        res = model.train(
            data=str(args["data_yaml"]), epochs=args["adaptive_epochs"], patience=args["tune_patience"], 
            batch=args["batch"], workers=args["workers"], project=str(args["tune_base"]), 
            name=f"gen_{args['gen']+1}", verbose=False, **args["current_params"]
        )
        
        actual_epochs = len(pd.read_csv(Path(res.save_dir) / "results.csv")) if (Path(res.save_dir) / "results.csv").exists() else args["adaptive_epochs"]
        mAP50 = res.results_dict.get('metrics/mAP50(B)', 0)
        mAP50_95 = res.results_dict.get('metrics/mAP50-95(B)', 0)
        fitness = ((0.1 * mAP50) + (0.9 * mAP50_95)) * 100
        
        queue.put({"success": True, "actual_epochs": actual_epochs, "mAP50": mAP50, "mAP50_95": mAP50_95, "fitness": fitness})
    except Exception:
        worker_logger.error(f"[Single Tune Run] 오류 발생: {traceback.format_exc()}")
        queue.put({"success": False, "error": traceback.format_exc()})
    finally:
        del model
        clear_vram()

def _tune_worker(args, queue):
    worker_logger = setup_logger()
    import time, traceback, multiprocessing
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    import optuna 
    
    start_time = time.time()
    result = {"success": False, "task": "tune", "error": "", "msg": "", "best_params": {}, "history": []}
    stop_event = args.get("stop_event")
    stop_handler = GracefulStopHandler(stop_event, worker_logger, result)
    worker_logger.info(f"[AutoML Worker] Optuna 튜닝 시작. 반복 횟수: {args['iterations']}")
    
    try:
        processed_dir, workspace_dir = Path(args["processed_dir"]), Path(args["workspace_dir"])
        model_name, iterations = args["model_name"], args["iterations"]
        tune_epochs = args.get("tune_epochs", 30)
        tune_patience = args.get("tune_patience", 5)
        
        tune_base = workspace_dir / "runs" / "tune_custom"
        if tune_base.exists(): shutil.rmtree(tune_base); worker_logger.debug("기존 tune_custom 폴더 삭제 완료")
        tune_base.mkdir(parents=True, exist_ok=True)
        
        img_map = {f.stem: f for f in sorted((processed_dir / "images").glob("*.jpg"))}
        lbl_map = {f.stem: f for f in sorted((processed_dir / "labels").glob("*.txt"))}
        paired = [(str(img_map[n]), str(lbl_map[n])) for n in img_map if n in lbl_map]
        worker_logger.debug(f"[AutoML Worker] 이미지-라벨 페어 개수: {len(paired)}")
        
        if len(paired) < 5: 
            result["error"] = "튜닝용 데이터 부족"
            worker_logger.error("데이터 부족으로 튜닝 종료"); queue.put(result); return
            
        tr, vl = train_test_split(paired, test_size=0.2)
        tr_txt, vl_txt = tune_base / "train.txt", tune_base / "val.txt"
        tr_txt.write_text("\n".join(str(Path(p[0]).resolve()) for p in tr))
        vl_txt.write_text("\n".join(str(Path(p[0]).resolve()) for p in vl))
        data_yaml = tune_base / "tune_data.yaml"
        data_yaml.write_text(f"train: {tr_txt.resolve()}\nval: {vl_txt.resolve()}\nnc: {len(args['class_names'])}\nnames: {args['class_names']}\n")
        
        search_history = []
        
        def objective(trial):
            if stop_handler.should_stop("Optuna 탐색 단계(Objective)"):
                worker_logger.info("🛑 사용자에 의한 탐색 취소. Optuna Study를 중단합니다.")
                trial.study.stop()
                raise optuna.exceptions.TrialPruned()

            current_params = {
                'box': round(trial.suggest_float('box', 0.1, 20.0), 4),
                'cls': round(trial.suggest_float('cls', 0.1, 10.0), 4),
                'dfl': round(trial.suggest_float('dfl', 0.1, 10.0), 4),
                'hsv_h': round(trial.suggest_float('hsv_h', 0.0, 0.1), 4),
                'hsv_s': round(trial.suggest_float('hsv_s', 0.0, 1.0), 4),
                'hsv_v': round(trial.suggest_float('hsv_v', 0.0, 1.0), 4),
                'degrees': round(trial.suggest_float('degrees', 0.0, 45.0), 4),
                'translate': round(trial.suggest_float('translate', 0.0, 0.5), 4),
                'scale': round(trial.suggest_float('scale', 0.0, 1.0), 4),
                'shear': round(trial.suggest_float('shear', 0.0, 30.0), 4),
                'flipud': round(trial.suggest_float('flipud', 0.0, 1.0), 4),
                'fliplr': round(trial.suggest_float('fliplr', 0.0, 1.0), 4),
                'mosaic': round(trial.suggest_float('mosaic', 0.0, 1.0), 4),
                'mixup': round(trial.suggest_float('mixup', 0.0, 1.0), 4),
                'copy_paste': round(trial.suggest_float('copy_paste', 0.0, 1.0), 4)
            }
            
            gen = trial.number
            worker_logger.debug(f"--- 튜닝 세대 {gen+1}/{iterations} --- 시작")
            
            adaptive_epochs = max(10, int((tune_epochs // 2) + (tune_epochs - tune_epochs // 2) * (gen / max(1, iterations - 1))))
            
            run_args = {
                "model_name": model_name, "data_yaml": data_yaml, "adaptive_epochs": adaptive_epochs,
                "tune_patience": tune_patience, "batch": args["batch"], "workers": args["workers"],
                "tune_base": tune_base, "gen": gen, "current_params": current_params,
                "webhook_url": args.get("webhook_url"), "noti_flags": args.get("noti_flags", {}),
                "stop_event": stop_event
            }
            
            run_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=_single_tune_run, args=(run_args, run_queue))
            p.start()
            
            run_res = None
            # [개선 1, 2] 데드락 방지 및 Subprocess 강제 종료
            while p.is_alive():
                if stop_event and stop_event.is_set():
                    worker_logger.warning("중지 요청 감지. 단일 튜닝 워커 강제 종료 시도.")
                    kill_process_tree(p.pid)
                    break
                try:
                    run_res = run_queue.get(timeout=0.5)
                    break
                except qlib.Empty:
                    continue
            p.join(timeout=2)
            
            # 큐에 남은 게 있을 수 있으니 한번 더 확인
            if run_res is None and not run_queue.empty():
                run_res = run_queue.get()
            
            if not run_res or not run_res.get("success"):
                err_msg = run_res.get('error') if run_res else '강제 종료됨'
                raise Exception(f"세대 {gen+1} 학습 중 오류: {err_msg}")
                
            actual_epochs, mAP50, mAP50_95, fitness = run_res["actual_epochs"], run_res["mAP50"], run_res["mAP50_95"], run_res["fitness"]
            
            if actual_epochs < adaptive_epochs and args.get("webhook_url") and args.get("noti_flags", {}).get("early_stop", False):
                worker_logger.info(f"세대 {gen+1} 조기 종료 감지됨 (Epoch: {actual_epochs})")
                send_discord_webhook(
                    webhook_url=args["webhook_url"],
                    title="🛑 [조기 종료 발동]",
                    description=f"Auto ML {gen+1}세대 - **{actual_epochs} Epoch**에서 학습이 조기 종료되었습니다.",
                    color=0xe74c3c,
                    sync=True # 동기식으로 확실하게 전송
                )

            search_history.append({
                "generation": gen + 1,
                "fitness": fitness,
                "mAP50": mAP50,
                "mAP50_95": mAP50_95,
                **current_params
            })
            
            worker_logger.info(f"세대 {gen+1}: Fitness={fitness:.2f} | Epochs실행: {adaptive_epochs}")
            return fitness

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.enqueue_trial(args["initial_params"])
        
        try:
            study.optimize(objective, n_trials=iterations)
        except optuna.exceptions.TrialPruned:
            worker_logger.info("탐색이 안전하게 조기 종료되었습니다. 지금까지의 결과를 반환합니다.")
        except Exception as e:
            if stop_handler.should_stop("최적화 진행 중 예외 발생"):
                worker_logger.info("탐색이 안전하게 조기 종료되었습니다. 지금까지의 결과를 반환합니다.")
            else:
                raise e

        if not search_history:
            result["error"] = "탐색 결과가 없습니다."
            return

        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        if args.get("webhook_url") and args.get("noti_flags", {}).get("tune", False):
            fields = [
                {"name": "최고 잠재력 점수", "value": f"`{best_score:.1f}점`", "inline": True},
                {"name": "최고 성능 세대", "value": f"{best_trial.number + 1}세대", "inline": True},
                {"name": "총 탐색 세대 개수", "value": f"{len(search_history)}개", "inline": False}
            ]
            send_discord_webhook(
                webhook_url=args["webhook_url"],
                title="🤖 [Auto ML] 하이퍼파라미터 탐색 완료",
                description="최적의 파라미터 조합 탐색이 완료되었습니다.",
                color=0x2ecc71,
                fields=fields,
                sync=True # 동기식으로 확실하게 전송
            )   
        result["best_params"] = best_params
        result["success"] = True
        result["history"] = search_history
        result["msg"] = f"✅ 맞춤형 파라미터 탐색 완료\n최고 잠재력 점수: {best_score:.1f}점 (세대: {best_trial.number + 1})\n총 {len(search_history)}세대 탐색 완료"
        worker_logger.info(f"[AutoML Worker] 튜닝 최종 완료. 소요시간: {time.time() - start_time:.1f}s")
        
    except Exception: 
        result["error"] = traceback.format_exc(); worker_logger.error(f"[AutoML Worker] Exception: {result['error']}")
    finally: 
        clear_vram(); queue.put(result)

def _auto_threshold_worker(args, queue):
    worker_logger = setup_logger()
    import time, traceback, numpy as np, gc
    from pathlib import Path
    from ultralytics import YOLO
    import torch
    import torchvision.ops as ops 
    
    start_time = time.time()
    result = {
        "success": False, "task": "threshold", "error": "",
        "best_conf": 0.25, "best_iou": 0.45, "best_acc": 0.0,
        "msg": "", "top5_params": [] 
    }
    
    stop_event = args.get("stop_event")
    stop_handler = GracefulStopHandler(stop_event, worker_logger, result)
    worker_logger.info(f"[AutoThreshold Worker] 임계값 자동 탐색 시작 (캐싱 모드). Model: {args['model_path']}")
    
    try:
        model_path, img_dir, lbl_dir = Path(args["model_path"]), Path(args["img_dir"]), Path(args["lbl_dir"])
        match_iou_thr = max(args.get("match_iou", 0.50), 0.1)
        agnostic, max_det = args.get("agnostic", True), args.get("max_det", 300)
        
        worker_logger.debug("모델 로딩 중...")
        model = YOLO(str(model_path))
        
        def calc_iou(b1, b2):
            ax1, ay1, ax2, ay2 = b1[0]-b1[2]/2, b1[1]-b1[3]/2, b1[0]+b1[2]/2, b1[1]+b1[3]/2
            bx1, by1, bx2, by2 = b2[0]-b2[2]/2, b2[1]-b2[3]/2, b2[0]+b2[2]/2, b2[1]+b2[3]/2
            ix, iy = max(0, min(ax2,bx2)-max(ax1,bx1)), max(0, min(ay2,by2)-max(ay1,by1))
            ia = ix*iy
            return ia/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-ia+1e-6)
        
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        total_imgs = len(img_files)
        worker_logger.debug(f"평가 대상 이미지 수: {total_imgs}장")
        
        if total_imgs == 0:
            raise ValueError("평가할 이미지가 없습니다.")
        
        gt_data = []
        for img_path in img_files:
            txt = lbl_dir / Path(img_path).with_suffix(".txt").name
            boxes = []
            if txt.exists():
                for line in txt.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append({
                            "class": int(parts[0]),
                            "box": [float(x) for x in parts[1:]]
                        })
            gt_data.append(boxes)
        
        worker_logger.info("⚡ 속도 최적화 및 VRAM 보호를 위한 일괄 추론 및 캐싱 진행 중...")
        
        cached_predictions = []
        BATCH_SIZE = 64 
        
        for batch_start in range(0, total_imgs, BATCH_SIZE):
            if stop_handler.should_stop("이미지 배치 캐싱 단계"):
                return
                
            batch_files = img_files[batch_start:batch_start + BATCH_SIZE]
            res_cached = model.predict(
                source=[str(p) for p in batch_files],
                conf=0.01,
                iou=0.99,
                max_det=500,
                agnostic_nms=agnostic, 
                verbose=False, 
                stream=True
            )
            
            for r in res_cached:
                cached_predictions.append({
                    "xyxy": r.boxes.xyxy.cpu().half() if len(r.boxes) > 0 else torch.empty((0, 4), dtype=torch.float16),
                    "xywhn": r.boxes.xywhn.cpu().half() if len(r.boxes) > 0 else torch.empty((0, 4), dtype=torch.float16),
                    "scores": r.boxes.conf.cpu().half() if len(r.boxes) > 0 else torch.empty(0, dtype=torch.float16),
                    "classes": r.boxes.cls.cpu().short() if len(r.boxes) > 0 else torch.empty(0, dtype=torch.int16)
                })
                
            del res_cached
            gc.collect()
            
        worker_logger.info(f"✅ 총 {len(cached_predictions)}장 캐싱 완료. 모델 메모리 해제.")
        del model
        clear_vram()
        
        def eval_yolo(test_conf, test_iou):
            total_tp = total_fp = total_fn = img_correct = 0
            for idx, cache in enumerate(cached_predictions):
                xyxy = cache["xyxy"].float()
                xywhn = cache["xywhn"].float()
                scores = cache["scores"].float()
                classes = cache["classes"].long()
                
                mask = scores >= test_conf
                f_xyxy = xyxy[mask]
                f_xywhn = xywhn[mask]
                f_scores = scores[mask]
                f_classes = classes[mask]
                
                if len(f_xyxy) > 0:
                    if agnostic:
                        keep = ops.nms(f_xyxy, f_scores, test_iou)
                    else:
                        keep = ops.batched_nms(f_xyxy, f_scores, f_classes, test_iou)
                    
                    if len(keep) > max_det:
                        keep = keep[:max_det]
                        
                    final_xywhn = f_xywhn[keep]
                    final_classes = f_classes[keep]
                    
                    pred_boxes = [{"class": int(cls.item()), "box": b.tolist()} for cls, b in zip(final_classes, final_xywhn)]
                else:
                    pred_boxes = []
                
                gt_boxes = gt_data[idx]
                matched_gt = set()
                tp = fp = 0
                
                for pb in pred_boxes:
                    best_iou, best_gt = 0.0, -1
                    for j, gb in enumerate(gt_boxes):
                        if j not in matched_gt and pb["class"] == gb["class"]:
                            iou = calc_iou(pb["box"], gb["box"])
                            if iou > best_iou:
                                best_iou, best_gt = iou, j
                    
                    if best_iou >= match_iou_thr:
                        tp += 1
                        matched_gt.add(best_gt)
                    else:
                        fp += 1
                
                fn = len(gt_boxes) - len(matched_gt)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
                if fp == 0 and fn == 0:
                    img_correct += 1
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            return img_correct, f1 * 100, precision, recall
        
        conf_1st = [0.05, 0.15, 0.25, 0.45, 0.65, 0.85, 0.95]
        iou_1st = [0.25, 0.40, 0.55, 0.70, 0.85]
        
        worker_logger.debug(f"1차 성근 탐색: {len(conf_1st)}×{len(iou_1st)}={len(conf_1st)*len(iou_1st)}가지")
        candidates = [] 
        
        for c in conf_1st:
            for i in iou_1st:
                if stop_handler.should_stop("1차 성근 탐색"):
                    return

                img_correct, f1, precision, recall = eval_yolo(c, i)
                candidates.append({"img_correct": img_correct, "f1": f1, "precision": precision, "recall": recall, "conf": c, "iou": i})
                worker_logger.info(f"⏳ [1차 탐색 중] Conf: {c:.2f}, IoU: {i:.2f} → 완벽 일치: {img_correct}장, F1: {f1:.2f}%")
        
        candidates_sorted = sorted(candidates, key=lambda x: (x["img_correct"], x["f1"]), reverse=True)
        top5 = candidates_sorted[:5]
        
        best_img_correct = top5[0]["img_correct"]
        best_f1 = top5[0]["f1"]
        best_conf = top5[0]["conf"]
        best_iou = top5[0]["iou"]
        
        worker_logger.debug("2차 정밀 탐색 시작 (Top-5 근처)")
        
        iteration = 0
        for rank, top_cand in enumerate(top5, 1):
            tc, ti = top_cand["conf"], top_cand["iou"]
            
            conf_range = 0.12 if 0.01 < tc < 0.99 else 0.08
            iou_range = 0.12 if 0.01 < ti < 0.99 else 0.08
            
            fine_c_start = max(0.01, tc - conf_range)
            fine_c_end = min(0.99, tc + conf_range)
            fine_i_start = max(0.01, ti - iou_range)
            fine_i_end = min(0.99, ti + iou_range)
            
            worker_logger.info(f"🔍 [2차 정밀 탐색 중] Top-{rank} 후보 주변 탐색: Conf({fine_c_start:.2f}~{fine_c_end:.2f}), IoU({fine_i_start:.2f}~{fine_i_end:.2f})")
            
            for c in np.arange(fine_c_start, fine_c_end + 0.02, 0.03):
                for i in np.arange(fine_i_start, fine_i_end + 0.02, 0.03):
                    iteration += 1
                    if stop_handler.check_every_n_iterations(iteration, interval=5, context="2차 정밀 탐색"):
                        return

                    img_correct, f1, precision, recall = eval_yolo(c, i)
                    is_new_best = ((img_correct > best_img_correct) or (img_correct == best_img_correct and f1 > best_f1))
                    
                    if is_new_best:
                        best_img_correct = img_correct
                        best_f1 = f1
                        best_conf = round(float(c), 2)
                        best_iou = round(float(i), 2)
                        worker_logger.info(f"✨ [최고점 갱신!] Conf={c:.3f}, IoU={i:.3f} → 완벽 일치={img_correct}장, F1={f1:.2f}%")
        
        result["success"] = True
        result["best_conf"] = best_conf
        result["best_iou"] = best_iou
        result["best_acc"] = round(float(best_f1), 1)
        result["best_img_correct"] = best_img_correct 
        
        result["top5_params"] = [
            {"rank": idx + 1, "conf": top["conf"], "iou": top["iou"], "img_correct": top["img_correct"], "f1": round(top["f1"], 2)}
            for idx, top in enumerate(top5)
        ]
        
        result["msg"] = (f"🎯 최적 스레숄드 탐색 완료\n"
                        f"• Conf={result['best_conf']}, IoU={result['best_iou']}\n"
                        f"• 완벽 매칭 이미지: {best_img_correct}장\n"
                        f"• 예상 F1-Score: {result['best_acc']}%")
        
        worker_logger.info(f"[AutoThreshold Worker] 탐색 완료. Conf: {result['best_conf']}, IoU: {result['best_iou']}, 완벽매칭: {best_img_correct}장, F1: {result['best_acc']}%")
        
    except Exception:
        result["error"] = traceback.format_exc()
        worker_logger.error(f"[AutoThreshold Worker] Exception: {result['error']}")
    finally:
        clear_vram()
        queue.put(result)

def _single_fold_run(args, queue):
    worker_logger = setup_logger()
    import traceback, pandas as pd
    from pathlib import Path
    from ultralytics import YOLO
    
    try:
        model = YOLO(args["model_name"])
        if args.get("webhook_url") and args.get("noti_flags", {}).get("epoch", False):
            interval = args.get("noti_flags", {}).get("epoch_interval", 100)
            model.add_callback("on_train_epoch_end", create_heartbeat_callback(args["webhook_url"], args["epochs"], interval))

        if args.get("stop_event"):
            for hook, cb in create_stop_callback(args["stop_event"]).items():
                model.add_callback(hook, cb)

        res = model.train(
            data=str(args["data_yaml"]), epochs=args["epochs"], patience=args["patience"], imgsz=args["imgsz"], 
            batch=args["batch"], workers=args["workers"], project=str(args["runs_dir"]), name=args["run_name"], 
            verbose=True, **args["aug"], 
            cls=args["loss"]["cls"], box=args["loss"]["box"], dfl=args["loss"]["dfl"]
        )
        
        actual_epochs = len(pd.read_csv(Path(res.save_dir) / "results.csv")) if (Path(res.save_dir) / "results.csv").exists() else args["epochs"]
        queue.put({"success": True, "actual_epochs": actual_epochs, "results_dict": res.results_dict, "save_dir": str(res.save_dir)})
    except Exception:
        worker_logger.error(f"[Single Fold Run] 오류 발생: {traceback.format_exc()}")
        queue.put({"success": False, "error": traceback.format_exc()})
    finally:
        del model
        clear_vram()

def _kfold_train_worker(args, queue):
    worker_logger = setup_logger()
    import json, shutil, numpy as np, time, multiprocessing; import pandas as pd; from sklearn.model_selection import KFold, train_test_split
    from pathlib import Path
    
    start_time = time.time(); result = {"success": False, "task": "train", "error": "", "msg": "", "best_model": ""}
    stop_event = args.get("stop_event")
    stop_handler = GracefulStopHandler(stop_event, worker_logger, result)
    worker_logger.info(f"[K-Fold Worker] 학습 시작. Folds: {args['num_folds']}, Model: {args['model_name']}, Epochs: {args['epochs']}")
    
    try:
        processed_dir, workspace_dir = Path(args["processed_dir"]), Path(args["workspace_dir"])
        kfold_base = workspace_dir / "kfold"; runs_dir = workspace_dir / "runs" / "kfold_train"
        img_map = {f.stem: f for f in sorted((processed_dir / "images").glob("*.jpg"))}
        lbl_map = {f.stem: f for f in sorted((processed_dir / "labels").glob("*.txt"))}
        paired = [(str(img_map[n]), str(lbl_map[n])) for n in img_map if n in lbl_map]
        worker_logger.debug(f"[K-Fold Worker] 매칭된 데이터 수: {len(paired)}개")
        if len(paired) < 5: result["error"] = "데이터 부족"; worker_logger.error("데이터 부족으로 종료"); queue.put(result); return
        train_val, test_files = train_test_split(paired, test_size=args["test_split"])
        if kfold_base.exists(): shutil.rmtree(kfold_base); worker_logger.debug("기존 kfold 폴더 초기화 완료")
        kfold_base.mkdir(parents=True, exist_ok=True)
        test_img_dir = kfold_base / "images" / "test"; test_lbl_dir = kfold_base / "labels" / "test"; test_img_dir.mkdir(parents=True, exist_ok=True); test_lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl in test_files: shutil.copy(img, test_img_dir); shutil.copy(lbl, test_lbl_dir)
        fold_metrics, fold_save_dirs = [], {}
        splits = [(train_test_split(train_val, test_size=args["test_split"]))] if args["num_folds"] == 1 else list(KFold(n_splits=args["num_folds"], shuffle=True).split(train_val))
        
        for fold, split_data in enumerate(splits):
            if stop_handler.should_stop(f"Fold {fold+1} 시작 전"):
                if not fold_metrics:
                    return
                else:
                    worker_logger.info("지금까지 완료된 Fold 중에서 최적의 모델을 선택하고 조기 종료합니다.")
                    break

            fold_num = fold + 1; fd = kfold_base / f"fold_{fold_num}"; fd.mkdir(exist_ok=True)
            worker_logger.info(f"--- Fold {fold_num}/{args['num_folds']} 학습 시작 ---")
            tr, vl = split_data if args["num_folds"] == 1 else (np.array(train_val)[split_data[0]], np.array(train_val)[split_data[1]])
            tr_txt, vl_txt = fd / "train.txt", fd / "val.txt"; tr_txt.write_text("\n".join(str(Path(p[0]).resolve()) for p in tr)); vl_txt.write_text("\n".join(str(Path(p[0]).resolve()) for p in vl))
            data_yaml = fd / "data.yaml"; data_yaml.write_text(f"train: {tr_txt.resolve()}\nval: {vl_txt.resolve()}\ntest: {test_img_dir.resolve()}\nnc: {len(args['class_names'])}\nnames: {args['class_names']}\n")
            
            run_name = f"fold_{fold_num}" if args["num_folds"] > 1 else "single_train"
            run_args = {
                "model_name": args["model_name"], "data_yaml": data_yaml, "epochs": args["epochs"], "patience": args.get("patience",100),
                "imgsz": args["imgsz"], "batch": args["batch"], "workers": args["workers"], "runs_dir": runs_dir, "run_name": run_name,
                "aug": args["aug"], "loss": args["loss"],
                "webhook_url": args.get("webhook_url"), "noti_flags": args.get("noti_flags", {}), "stop_event": stop_event
            }
            
            run_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=_single_fold_run, args=(run_args, run_queue))
            p.start()
            
            run_res = None
            # [개선 1, 2] 데드락 방지 및 Subprocess 강제 종료
            while p.is_alive():
                if stop_event and stop_event.is_set():
                    worker_logger.warning("중지 요청 감지. 단일 폴드 워커 강제 종료 시도.")
                    kill_process_tree(p.pid)
                    break
                try:
                    run_res = run_queue.get(timeout=0.5)
                    break
                except qlib.Empty:
                    continue
            p.join(timeout=2)
            
            if run_res is None and not run_queue.empty():
                run_res = run_queue.get()
            
            if not run_res or not run_res.get("success"):
                err_msg = run_res.get('error') if run_res else '강제 종료됨'
                raise Exception(f"Fold {fold_num} 학습 중 오류: {err_msg}")
                
            actual_epochs, res_dict, save_dir = run_res["actual_epochs"], run_res["results_dict"], run_res["save_dir"]
            
            if actual_epochs < args["epochs"]:
                worker_logger.info(f"Fold {fold_num} 조기 종료 감지. (Epoch: {actual_epochs})")
                if args.get("webhook_url") and args.get("noti_flags", {}).get("early_stop", False):
                    send_discord_webhook(
                        webhook_url=args["webhook_url"],
                        title="🛑 [조기 종료 발동]",
                        description=f"Fold {fold_num} - **{actual_epochs} Epoch**에서 학습이 조기 종료되었습니다.",
                        color=0xe74c3c,
                        sync=True # 동기식으로 확실하게 전송
                    )

            fold_metrics.append(res_dict); fold_save_dirs[fold_num] = save_dir
            
            if args.get("webhook_url") and args.get("noti_flags", {}).get("fold", False):
                mAP = res_dict.get('metrics/mAP50-95(B)', 0)
                send_discord_webhook(
                    webhook_url=args["webhook_url"],
                    title=f"📍 [K-Fold] Fold {fold_num} 완료",
                    description=f"검증 mAP50-95: **{mAP:.4f}**",
                    color=0x2ecc71,
                    sync=True # 동기식으로 확실하게 전송
                )

        if fold_metrics:
            best_n = max(range(len(fold_metrics)), key=lambda i: (fold_metrics[i].get(args["best_metric"], 0), fold_metrics[i].get(args["second_metric"], 0))) + 1
            worker_logger.info(f"[K-Fold Worker] 모든 Fold 종료. 최우수 Fold: {best_n}")
            src = Path(fold_save_dirs[best_n]) / "weights" / "best.pt"; dst = kfold_base / "best_model.pt"
            if src.exists():
                shutil.copy(src, dst); (kfold_base / "best_model_source.txt").write_text(str(src), encoding="utf-8")
                result["success"] = True; result["msg"] = f"✅ 학습 완료\n최우수 Fold: {best_n}"; result["best_model"] = str(dst); result["original_model_path"] = str(src)
                summary = [{"Fold": i + 1, "mAP50": fm.get("metrics/mAP50(B)", 0), "mAP50-95": fm.get("metrics/mAP50-95(B)", 0), "Precision": fm.get("metrics/precision(B)", 0), "Recall": fm.get("metrics/recall(B)", 0), "Fitness": (0.1 * fm.get("metrics/mAP50(B)", 0)) + (0.9 * fm.get("metrics/mAP50-95(B)", 0))} for i, fm in enumerate(fold_metrics)]
                summary.append({"Fold": "Average", "mAP50": sum(m["mAP50"] for m in summary)/len(summary), "mAP50-95": sum(m["mAP50-95"] for m in summary)/len(summary), "Precision": sum(m["Precision"] for m in summary)/len(summary), "Recall": sum(m["Recall"] for m in summary)/len(summary), "Fitness": sum(m["Fitness"] for m in summary)/len(summary)})
                result["metrics_summary"] = summary; result["best_fold"] = best_n
            else: result["error"] = "가중치 파일이 없습니다."; worker_logger.error("Best 가중치 파일 탐색 실패")
        else: result["error"] = "학습 실패"; worker_logger.error("fold_metrics가 비어있음")
    except Exception: result["error"] = traceback.format_exc(); worker_logger.error(f"[K-Fold Worker] Exception: {result['error']}")
    finally: clear_vram(); queue.put(result)

def _retrain_worker(args, queue):
    worker_logger = setup_logger()
    import yaml as _yaml, shutil, time; import pandas as pd; from ultralytics import YOLO; from pathlib import Path
    start_time = time.time(); result = {"success": False, "task": "retrain", "error": "", "msg": "", "save_dir": "", "model_path": ""}
    worker_logger.info(f"[Retrain Worker] 하드 재학습 시작. Base Model: {args['rt_base_model']}")
    try:
        p = args; retrain_ds = Path(p["workspace_dir"]) / "retrain"; runs_dir = Path(p["workspace_dir"]) / "runs" / "retrain_train"
        new_img, new_lbl = retrain_ds / "images" / "train", retrain_ds / "labels" / "train"
        if retrain_ds.exists(): shutil.rmtree(retrain_ds)
        new_img.mkdir(parents=True, exist_ok=True); new_lbl.mkdir(parents=True, exist_ok=True)
        files = iter_supported_images(p["rt_hard_dir"])
        worker_logger.debug(f"[Retrain Worker] 추출된 오답 이미지 수: {len(files)}")
        if not files: result["error"] = "오답 이미지가 없습니다."; queue.put(result); return
        for img_file in files:
            shutil.copy(img_file, new_img / img_file.name)
            src_txt = Path(p["rt_orig_labels"]) / img_file.with_suffix(".txt").name; dst_txt = new_lbl / img_file.with_suffix(".txt").name
            shutil.copy(src_txt, dst_txt) if src_txt.exists() else dst_txt.touch()
        yaml_rt = retrain_ds / "retrain_data.yaml"
        yaml_rt.write_text(_yaml.dump({"path": retrain_ds.resolve().as_posix(), "train": "images/train", "val": "images/train", "nc": len(p["class_names"]), "names": p["class_names"]}, sort_keys=False), encoding="utf-8")
        model = YOLO(str(p["rt_base_model"]))
        
        if p.get("webhook_url") and p.get("noti_flags", {}).get("epoch", False):
            interval = p.get("noti_flags", {}).get("epoch_interval", 10) 
            model.add_callback("on_train_epoch_end", create_heartbeat_callback(p["webhook_url"], p["rt_epochs"], interval))

        if p.get("stop_event"):
            for hook, cb in create_stop_callback(p["stop_event"]).items():
                model.add_callback(hook, cb)

        res = model.train(data=str(yaml_rt), epochs=p["rt_epochs"], imgsz=p["imgsz"], batch=p["rt_batch"], project=str(runs_dir), name=p["rt_run_name"], exist_ok=True, hsv_h=p["rt_h"], hsv_s=p["rt_s"], hsv_v=p["rt_v"], flipud=p["rt_flipud"], fliplr=p["rt_fliplr"], mosaic=p["rt_mosaic"], mixup=p["rt_mix"], copy_paste=p["rt_cp"], cls=p["rt_cls"], box=p["rt_box"], verbose=True)
        
        actual_epochs = len(pd.read_csv(Path(res.save_dir) / "results.csv")) if (Path(res.save_dir) / "results.csv").exists() else p["rt_epochs"]
        if actual_epochs < p["rt_epochs"] and p.get("webhook_url") and p.get("noti_flags", {}).get("early_stop", False):
            worker_logger.info(f"[Retrain Worker] 조기 종료됨 (Epoch: {actual_epochs})")
            send_discord_webhook(
                webhook_url=p["webhook_url"],
                title="🛑 [조기 종료 발동]",
                description=f"재학습이 **{actual_epochs} Epoch**에서 조기 종료되었습니다.",
                color=0xe74c3c,
                sync=True # 동기식으로 확실하게 전송
            )

        trained_model_path = Path(res.save_dir) / "weights" / "best.pt"; del model
        result["success"] = True; result["msg"] = f"✅ 재학습 완료!\n저장: {res.save_dir}"; result["save_dir"] = str(res.save_dir); result["model_path"] = str(trained_model_path)
        worker_logger.info("[Retrain Worker] 하드 재학습 모두 완료됨")
    except Exception: result["error"] = traceback.format_exc(); worker_logger.error(f"[Retrain Worker] Exception: {result['error']}")
    finally: clear_vram(); queue.put(result)

# =========================================================
# [개선 3] 완벽한 VRAM 격리를 위한 Eval / Measure Worker 분리
# =========================================================
def _eval_worker(args, queue):
    worker_logger = setup_logger()
    import shutil
    from ultralytics import YOLO
    result = {"success": False, "task": "eval", "error": ""}
    stop_event = args.get("stop_event")
    try:
        c = args; eval_project_dir = Path(c["workspace_dir"]) / "runs" / "eval"
        relabel_dir = eval_project_dir / f"{c['eval_run_name']}_needs_relabel"
        if c["save_relabel"]:
            if relabel_dir.exists(): shutil.rmtree(relabel_dir)
            relabel_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(c["eval_model_path"]))
        res = model.predict(source=str(c["eval_source"]), save=True, conf=c["eval_conf"], iou=c["eval_iou"], max_det=c["max_det"], project=str(eval_project_dir), name=c["eval_run_name"], agnostic_nms=c["agnostic_nms"], exist_ok=True)
        
        def calc_iou(b1, b2):
            ax1,ay1,ax2,ay2 = b1[0]-b1[2]/2, b1[1]-b1[3]/2, b1[0]+b1[2]/2, b1[1]+b1[3]/2
            bx1,by1,bx2,by2 = b2[0]-b2[2]/2, b2[1]-b2[3]/2, b2[0]+b2[2]/2, b2[1]+b2[3]/2
            ix, iy = max(0, min(ax2,bx2)-max(ax1,bx1)), max(0, min(ay2,by2)-max(ay1,by1))
            ia = ix*iy; return ia/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-ia+1e-6)
            
        rows, wrong, wrong_imgs, all_imgs = [], 0, [], []; total_tp, total_fp, total_fn = 0, 0, 0
        for r in res:
            if stop_event and stop_event.is_set(): raise Exception("사용자 취소됨")
            img_name = Path(r.path).name; current_img_path = str(eval_project_dir / c["eval_run_name"] / img_name); all_imgs.append(current_img_path)
            pred_boxes = [{"class": int(cls.item()), "box": b.tolist()} for cls, b in zip(r.boxes.cls, r.boxes.xywhn)] if len(r.boxes) > 0 else []
            gt_boxes = []; txt = Path(c["gt_labels_path"]) / Path(img_name).with_suffix(".txt").name
            if txt.exists():
                for line in txt.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5: gt_boxes.append({"class": int(parts[0]), "box": [float(x) for x in parts[1:]]})
            matched_gt = set(); tp = fp = 0
            for pb in pred_boxes:
                best_iou, best_gt = 0.0, -1
                for j, gb in enumerate(gt_boxes):
                    if j not in matched_gt and pb["class"] == gb["class"]:
                        iou = calc_iou(pb["box"], gb["box"])
                        if iou > best_iou: best_iou, best_gt = iou, j
                if best_iou >= max(c.get("match_iou", 0.5), 0.5): tp += 1; matched_gt.add(best_gt)
                else: fp += 1
            fn = len(gt_boxes) - len(matched_gt); total_tp += tp; total_fp += fp; total_fn += fn
            correct = (fp == 0 and fn == 0); reasons = []
            if fn > 0: reasons.append(f"미검 {fn}개")
            if fp > 0: reasons.append(f"과검 {fp}개")
            if not correct:
                wrong += 1; wrong_imgs.append(current_img_path)
                if c["save_relabel"]: shutil.copy(r.path, relabel_dir / img_name)
            rows.append({"파일명": img_name, "상태": "✅ 정상" if correct else "❌ 오답", "예측 수": len(pred_boxes), "정답 수": len(gt_boxes), "사유": ", ".join(reasons) if reasons else "정확히 일치"})
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        stats = {"total": len(rows), "correct": len(rows) - wrong, "wrong": wrong, "acc": f1_score * 100, "precision": precision * 100, "recall": recall * 100, "f1_score": f1_score * 100, "relabel_dir": str(relabel_dir) if c["save_relabel"] else ""}
        worker_logger.info(f"[Eval Worker] 평가 완료. F1-Score: {f1_score*100:.2f}%, 오답: {wrong}건")
        
        result.update({"success": True, "rows": rows, "wrong_imgs": wrong_imgs, "all_imgs": all_imgs, "stats": stats})
    except Exception: result["error"] = traceback.format_exc(); worker_logger.error("[Eval Worker] 에러 발생", exc_info=True)
    finally: clear_vram(); queue.put(result)

def _tab4_eval_worker(args, queue):
    worker_logger = setup_logger()
    try:
        from ultralytics import YOLO
        c = args; model = YOLO(str(c["retrained_model"]))
        val_metrics = model.val(data=str(c["yaml_path"]), conf=c["eval_conf"], iou=c["eval_iou"], max_det=c["max_det"], project=str(Path(c["workspace_dir"]) / "runs" / "eval"), name=c["eval_run_name"] + "_val")
        pr_curve = str(Path(val_metrics.save_dir) / "PR_curve.png")
        
        eval_config = c.copy(); eval_config["eval_model_path"] = c["retrained_model"]; eval_config["eval_run_name"] = c["eval_run_name"] + "_final_eval"; eval_config["save_relabel"] = False
        del model; clear_vram()
        
        import queue as local_q
        q2 = local_q.Queue()
        _eval_worker(eval_config, q2)
        res = q2.get()
        if res.get("success"):
            res["pr_curve"] = pr_curve
            queue.put(res)
        else:
            queue.put(res)
    except Exception: 
        worker_logger.error("[Tab4FinalEval Worker] 에러 발생", exc_info=True)
        queue.put({"success": False, "error": traceback.format_exc()})
    finally: clear_vram()

def _measure_worker(args, queue):
    worker_logger = setup_logger()
    from ultralytics import YOLO
    import math
    COLOR_MAP = {"노란색 (Yellow)": (0, 255, 255), "초록색 (Green)": (0, 255, 0), "빨간색 (Red)": (0, 0, 255), "파란색 (Blue)": (255, 0, 0), "청록색 (Cyan)": (255, 255, 0), "자주색 (Magenta)": (255, 0, 255), "흰색 (White)": (255, 255, 255)}
    result = {"success": False, "task": "measure", "error": ""}
    stop_event = args.get("stop_event")
    
    try:
        c = args; dist_source = Path(c["dist_source"])
        is_edge, is_knn = c["measure_method"] == "테두리 최단거리 (Edge)", c["measure_method"] == "가장 가까운 N개 이웃 (방향 무관)"
        c1 = COLOR_MAP.get(c.get("color1", "노란색 (Yellow)"), (0, 255, 255)); c2 = COLOR_MAP.get(c.get("color2", "청록색 (Cyan)"), (255, 255, 0))
        dist_run_name = "ok_edge_distance" if is_edge else "ok_knn_distance" if is_knn else "ok_euclidean_distance"
        dist_col_name = "수평/수직 테두리 거리(px)" if is_edge else "최단 중심점 거리_N개(px)" if is_knn else "중심점 간 유클리드 거리(px)"
        base_dir = Path(c["workspace_dir"]) / "runs" / "measure"; base_dir.mkdir(parents=True, exist_ok=True); folder_idx = 1
        while (base_dir / f"{dist_run_name}_{folder_idx:02d}").exists(): folder_idx += 1
        final_save_dir = base_dir / f"{dist_run_name}_{folder_idx:02d}"; final_save_dir.mkdir()
        
        model = YOLO(str(c["dist_model_path"]))
        img_files = iter_supported_images(dist_source); total_imgs = len(img_files)
        results = model.predict(source=str(dist_source), save=False, conf=c["dist_conf"], iou=c["dist_iou"], max_det=c["dist_max_det"], agnostic_nms=c["dist_agnostic"], stream=True)
        
        image_pairs = []; processed = 0; all_data = []
        thresholds = [float(x.strip()) for x in c["edge_thresholds"].split(",") if x.strip()] if is_edge else [60.0]
        def get_cat(d, thr):
            if d < thr[0]: return f"유형 1 ( < {thr[0]} )"
            for i in range(len(thr)-1):
                if thr[i] <= d < thr[i+1]: return f"유형 {i+2} ( {thr[i]}~{thr[i+1]} )"
            return f"유형 {len(thr)+1} ( >= {thr[-1]} )"

        for r in results:
            if stop_event and stop_event.is_set(): raise Exception("사용자 취소됨")
            processed += 1
            if processed % 10 == 0: queue.put({"type": "progress", "val": int(processed / max(total_imgs, 1) * 100)})
            img_name = Path(r.path).name; result_img_path = final_save_dir / img_name
            detected = [r.names[int(cls_id)].lower() for cls_id in r.boxes.cls]
            if c["skip_ng"] and "ng" in detected: continue
            num_objects = len(r.boxes)
            if c["drop_odd_lowest"] and num_objects % 2 != 0 and num_objects > 0:
                r = r[[i for i in range(num_objects) if i != int(r.boxes.conf.argmin().item())]]; num_objects = len(r.boxes)
            plotted_img = r.plot(line_width=1, conf=False, labels=False); measured_distances = []; worst_conf = r.boxes.conf.min().item() if num_objects > 0 else 0.0
            if num_objects >= 2:
                boxes = r.boxes.xyxy.cpu().numpy()
                for i, b1 in enumerate(boxes):
                    min_x = min_y = float("inf"); r_pts = b_pts = None; r_dist = b_dist = 0
                    cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2; knn_cands = []
                    for j, b2 in enumerate(boxes):
                        if i == j: continue
                        ox, oy = min(b1[2], b2[2]) - max(b1[0], b2[0]), min(b1[3], b2[3]) - max(b1[1], b2[1])
                        cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2; dist = math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                        if is_knn: knn_cands.append((dist, (int(cx1), int(cy1)), (int(cx2), int(cy2))))
                        elif is_edge:
                            if oy > 0 and b2[0] >= b1[2] and (b2[0] - b1[2]) < min_x: min_x = r_dist = b2[0] - b1[2]; y_cen = int((max(b1[1], b2[1]) + min(b1[3], b2[3])) / 2); r_pts = ((int(b1[2]), y_cen), (int(b2[0]), y_cen))
                            if ox > 0 and b2[1] >= b1[3] and (b2[1] - b1[3]) < min_y: min_y = b_dist = b2[1] - b1[3]; x_cen = int((max(b1[0], b2[0]) + min(b1[2], b2[2])) / 2); b_pts = ((x_cen, int(b1[3])), (x_cen, int(b2[1])))
                        else:
                            if oy > 0 and cx2 > cx1 and dist < min_x: min_x = r_dist = dist; r_pts = ((int(cx1), int(cy1)), (int(cx2), int(cy2)))
                            if ox > 0 and cy2 > cy1 and dist < min_y: min_y = b_dist = dist; b_pts = ((int(cx1), int(cy1)), (int(cx2), int(cy2)))
                    if is_knn:
                        for d, pt1, pt2 in sorted(knn_cands, key=lambda x: x[0])[:c["n_neighbors"]]:
                            cv2.line(plotted_img, pt1, pt2, c1, 2); cv2.putText(plotted_img, f"{d:.1f}", (int((pt1[0]+pt2[0])/2), int((pt1[1]+pt2[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c1, 2); measured_distances.append(round(d, 1))
                    else:
                        if r_pts: cv2.line(plotted_img, r_pts[0], r_pts[1], c1, 2); cv2.putText(plotted_img, f"{r_dist:.1f}", (int((r_pts[0][0]+r_pts[1][0])/2), int((r_pts[0][1]+r_pts[1][1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c1, 2); measured_distances.append(round(r_dist, 1))
                        if b_pts: cv2.line(plotted_img, b_pts[0], b_pts[1], c2, 2); cv2.putText(plotted_img, f"{b_dist:.1f}", (int((b_pts[0][0]+b_pts[1][0])/2), int((b_pts[0][1]+b_pts[1][1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c2, 2); measured_distances.append(round(b_dist, 1))
                
                is_success, im_buf_arr = cv2.imencode(".jpg", plotted_img)
                if is_success: im_buf_arr.tofile(str(result_img_path))
                
            dstr = ", ".join([str(d) for d in sorted(list(set(measured_distances)))]) if measured_distances else "측정된 선 없음"
            image_pairs.append({"파일명": img_name, "탐지 수": f"{num_objects}개", "최저 신뢰도": round(worst_conf, 2), dist_col_name: dstr, "_img_path": str(result_img_path)})
            if measured_distances:
                for d in measured_distances: all_data.append({"파일명": img_name, "구분": get_cat(d, thresholds) if is_edge else "유클리드", "거리(px)": d})
            
        df_export = pd.DataFrame([{k:v for k,v in item.items() if k!="_img_path"} for item in image_pairs])
        df_export.to_csv(final_save_dir / f"{dist_run_name}_results.csv", index=False, encoding="utf-8-sig")
        df_parsed = pd.DataFrame(all_data); outliers_list = []
        if not df_parsed.empty:
            stats_list = []  
            for cat in df_parsed["구분"].unique():
                subset = df_parsed[df_parsed["구분"] == cat]
                Q1, Q3 = subset["거리(px)"].quantile(0.25), subset["거리(px)"].quantile(0.75); IQR = Q3 - Q1; lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                outs = subset[(subset["거리(px)"] < lo) | (subset["거리(px)"] > hi)].copy()
                if not outs.empty: outs["하한"], outs["상한"] = round(lo, 2), round(hi, 2); outliers_list.append(outs)
                stats_list.append({"구분(유형)": cat, "데이터 수(Count)": len(subset), "평균(Mean)": round(subset["거리(px)"].mean(), 2), "표준편차(Std)": round(subset["거리(px)"].std(), 2) if len(subset) > 1 else 0.0, "최소값(Min)": round(subset["거리(px)"].min(), 2), "1사분위(Q1)": round(Q1, 2), "중앙값(Median)": round(subset["거리(px)"].median(), 2), "3사분위(Q3)": round(Q3, 2), "최대값(Max)": round(subset["거리(px)"].max(), 2), "IQR": round(IQR, 2), "정상 하한값(Lower Bounds)": round(lo, 2), "정상 상한값(Upper Bounds)": round(hi, 2), "이상치 개수": len(outs)})
            pd.DataFrame(stats_list).to_csv(final_save_dir / f"{dist_run_name}_statistics.csv", index=False, encoding="utf-8-sig")
            if outliers_list: pd.concat(outliers_list, ignore_index=True).to_csv(final_save_dir / f"{dist_run_name}_outliers.csv", index=False, encoding="utf-8-sig")
            
        df_outliers_dict = pd.concat(outliers_list, ignore_index=True).to_dict() if outliers_list else {}
        worker_logger.info(f"[Measure Worker] 측정 완료. 결과 건수: {len(df_export)}, 이상치: {len(outliers_list)}")
        result.update({"success": True, "type": "result", "df_export": df_export.to_dict(), "df_parsed": df_parsed.to_dict(), "df_outliers": df_outliers_dict, "image_pairs": image_pairs})
    except Exception: result["error"] = traceback.format_exc(); worker_logger.error("[Measure Worker] 에러 발생", exc_info=True)
    finally: clear_vram(); queue.put(result)

# =========================================================
# QThread Wrappers (UI와 통신하기 위해 Process를 관장)
# =========================================================
