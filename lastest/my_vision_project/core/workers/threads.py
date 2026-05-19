from ...utils import *
from ...core import *
from .training import _eval_worker, _tab4_eval_worker, _measure_worker

# =========================================================
# QThread Wrappers (UI와 통신하기 위해 Process를 관장)
# =========================================================
class ProcessMonitorThread(QThread):
    finished_ok = pyqtSignal(dict); error = pyqtSignal(str)
    def __init__(self, queue, process, stop_event=None): super().__init__(); self.queue = queue; self.process = process; self.stop_event = stop_event
    def run(self):
        logger.debug(f"[Monitor Thread] 모니터링 시작 (PID: {self.process.pid})")
        result = None
        while self.process.is_alive():
            if self.stop_event and self.stop_event.is_set():
                logger.warning("UI 쓰레드에서 중지 신호 감지 -> 워커 프로세스 강제 종료 시도")
                kill_process_tree(self.process.pid); break
            try: result = self.queue.get(timeout=0.5); break
            except qlib.Empty: continue
        self.process.join(timeout=2)
        if result is None:
            try: result = self.queue.get(timeout=0.5)
            except qlib.Empty: pass
        if result: self.finished_ok.emit(result)
        else: self.error.emit(f"프로세스 비정상 종료 (Exit Code: {self.process.exitcode})")

class EvalThread(QThread):
    finished_ok = pyqtSignal(pd.DataFrame, list, list, dict); error = pyqtSignal(str)
    def __init__(self, config): super().__init__(); self.config = config
    def run(self):
        logger.info(f"[Eval Thread] QThread 시작 (Process 분리). 대상 모델: {self.config['eval_model_path']}")
        run_queue = multiprocessing.Queue(); p = multiprocessing.Process(target=_eval_worker, args=(self.config, run_queue)); p.start()
        res = None; stop_event = self.config.get("stop_event")
        while p.is_alive():
            if stop_event and stop_event.is_set(): kill_process_tree(p.pid); break
            try: res = run_queue.get(timeout=0.5); break
            except qlib.Empty: continue
        p.join(timeout=2)
        if not res and not run_queue.empty(): res = run_queue.get()
        if res and res.get("success"): self.finished_ok.emit(pd.DataFrame(res["rows"]).sort_values("상태"), res["wrong_imgs"], res["all_imgs"], res["stats"])
        else: self.error.emit(res.get("error") if res else "프로세스 강제 종료됨")

class Tab4FinalEvalThread(QThread):
    finished_ok = pyqtSignal(pd.DataFrame, list, list, dict, str); error = pyqtSignal(str)
    def __init__(self, config): super().__init__(); self.config = config
    def run(self):
        logger.info("[Tab4FinalEval Thread] QThread 시작 (Process 분리)")
        run_queue = multiprocessing.Queue(); p = multiprocessing.Process(target=_tab4_eval_worker, args=(self.config, run_queue)); p.start()
        res = None; stop_event = self.config.get("stop_event")
        while p.is_alive():
            if stop_event and stop_event.is_set(): kill_process_tree(p.pid); break
            try: res = run_queue.get(timeout=0.5); break
            except qlib.Empty: continue
        p.join(timeout=2)
        if not res and not run_queue.empty(): res = run_queue.get()
        if res and res.get("success"): self.finished_ok.emit(pd.DataFrame(res["rows"]).sort_values("상태"), res["wrong_imgs"], res["all_imgs"], res["stats"], res["pr_curve"])
        else: self.error.emit(res.get("error") if res else "프로세스 강제 종료됨")

class MeasureThread(QThread):
    progress = pyqtSignal(int); finished_ok = pyqtSignal(pd.DataFrame, pd.DataFrame, pd.DataFrame, list); error = pyqtSignal(str)
    def __init__(self, config): super().__init__(); self.config = config
    def run(self):
        logger.info("[Measure Thread] QThread 시작 (Process 분리)")
        run_queue = multiprocessing.Queue(); p = multiprocessing.Process(target=_measure_worker, args=(self.config, run_queue)); p.start()
        res = None; stop_event = self.config.get("stop_event")
        while p.is_alive():
            if stop_event and stop_event.is_set(): kill_process_tree(p.pid); break
            try:
                msg = run_queue.get(timeout=0.5)
                if msg.get("type") == "progress": self.progress.emit(msg["val"])
                elif msg.get("type") == "result": res = msg; break
            except qlib.Empty: continue
        p.join(timeout=2)
        if not res and not run_queue.empty():
            msg = run_queue.get()
            if msg.get("type") == "result": res = msg
        if res and res.get("success"): self.finished_ok.emit(pd.DataFrame(res["df_export"]), pd.DataFrame(res["df_parsed"]), pd.DataFrame(res["df_outliers"]), res["image_pairs"])
        else: self.error.emit(res.get("error") if res else "프로세스 강제 종료됨")

class PreprocessThread(QThread):
    progress = pyqtSignal(int); log_msg = pyqtSignal(str); finished_ok = pyqtSignal(int); error = pyqtSignal(str)
    def __init__(self, config): super().__init__(); self.config = config
    def run(self):
        logger.info("[Preprocess Thread] 데이터 전처리 쓰레드 시작")
        try:
            from PIL import Image as PILImage, ImageOps, ImageDraw
            c = self.config; label_dir, image_dir = Path(c["base_dir"]) / "data", Path(c["base_dir"]) / "image"
            out_img, out_lbl, out_preview = Path(c["processed_dir"]) / "images", Path(c["processed_dir"]) / "labels", Path(c["processed_dir"]) / "preview"
            label_files = list(label_dir.glob("*.json")) + list(label_dir.glob("*.txt"))
            logger.debug(f"전처리 대상 라벨 파일 수: {len(label_files)}")
            crop_x, crop_y, crop_w, crop_h = c["manual_crop"]
            
            dim_cache = {}

            if c["use_auto_crop"]:
                logger.debug("오토 크롭 설정 적용됨, 좌표 계산 중...")
                min_x = min_y = float("inf"); max_x = max_y = 0.0
                for lf in label_files:
                    if lf.suffix == ".json":
                        for item in json.loads(lf.read_text(encoding="utf-8")).get("area", []):
                            if isinstance(item, dict) and len(item) == 1 and len(list(item.values())[0]) == 4:
                                x, y, w, h = list(item.values())[0]; min_x, min_y = min(min_x, x), min(min_y, y); max_x, max_y = max(max_x, x + w), max(max_y, y + h)
                    elif lf.suffix == ".txt":
                        img_path = find_image_for_label(image_dir, lf)
                        if img_path:
                            try:
                                if img_path.name not in dim_cache:
                                    with PILImage.open(img_path) as img:
                                        if c["use_exif"]: img = ImageOps.exif_transpose(img)
                                        dim_cache[img_path.name] = img.size
                                
                                iw, ih = dim_cache[img_path.name]

                                for line in lf.read_text(encoding="utf-8").strip().splitlines():
                                    parts = line.strip().split()
                                    if len(parts) == 5:
                                        w, h = float(parts[3]) * iw, float(parts[4]) * ih; x, y = (float(parts[1]) * iw) - (w / 2), (float(parts[2]) * ih) - (h / 2)
                                        min_x, min_y = min(min_x, x), min(min_y, y); max_x, max_y = max(max_x, x + w), max(max_y, y + h)
                            except: pass
                if min_x != float("inf"):
                    crop_x, crop_y = max(0, int(min_x - c["margin"])), max(0, int(min_y - c["margin"]))
                    crop_w, crop_h = int((max_x + c["margin"]) - crop_x), int((max_y + c["margin"]) - crop_y)
            self.log_msg.emit(f"✂️ 크롭 영역 확정 → X={crop_x}, Y={crop_y}, W={crop_w}, H={crop_h}")
            logger.info(f"크롭 설정 완료: X={crop_x}, Y={crop_y}, W={crop_w}, H={crop_h}")
            
            if c["clean_old"] and Path(c["processed_dir"]).exists(): shutil.rmtree(c["processed_dir"]); logger.debug("기존 전처리 폴더 삭제됨")
            out_img.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True); out_preview.mkdir(parents=True, exist_ok=True)
            ok_count = 0
            for i, lf in enumerate(label_files):
                img_id = None; boxes_abs = []
                if lf.suffix == ".json":
                    ann = json.loads(lf.read_text(encoding="utf-8")); img_id = ann.get("id")
                    if not img_id or not (image_dir / img_id).exists(): continue
                    for item in ann.get("area", []):
                        if isinstance(item, dict) and len(item) == 1:
                            label, bbox = list(item.keys())[0], list(item.values())[0]
                            if label in c["class_map"] and len(bbox) == 4: boxes_abs.append((c["class_map"][label], bbox))
                elif lf.suffix == ".txt":
                    temp = find_image_for_label(image_dir, lf)
                    if temp: img_id = temp.name
                    if not img_id: continue
                try:
                    img = PILImage.open(image_dir / img_id)
                    if c["use_exif"]: img = ImageOps.exif_transpose(img)
                    
                    iw, ih = dim_cache.get(img_id, img.size)

                    acw, ach = min(crop_w, iw - crop_x), min(crop_h, ih - crop_y)
                    if acw <= 0 or ach <= 0: continue
                    if lf.suffix == ".txt":
                        for line in lf.read_text(encoding="utf-8").strip().splitlines():
                            parts = line.strip().split()
                            if len(parts) == 5:
                                w, h = float(parts[3]) * iw, float(parts[4]) * ih
                                boxes_abs.append((int(parts[0]), [(float(parts[1]) * iw) - (w / 2), (float(parts[2]) * ih) - (h / 2), w, h]))
                    labels = []
                    for cid, (xo, yo, wo, ho) in boxes_abs:
                        nx, ny = xo - crop_x, yo - crop_y
                        if nx + wo > 0 and ny + ho > 0 and nx < acw and ny < ach:
                            fx1, fy1, fx2, fy2 = max(0, nx), max(0, ny), min(acw, nx + wo), min(ach, ny + ho)
                            fw, fh = fx2 - fx1, fy2 - fy1; labels.append(f"{cid} {(fx1 + fw / 2) / acw:.6f} {(fy1 + fh / 2) / ach:.6f} {fw/acw:.6f} {fh/ach:.6f}")
                    (out_lbl / Path(img_id).with_suffix(".txt").name).write_text("\n".join(labels), encoding="utf-8")
                    cropped_img = img.crop((crop_x, crop_y, crop_x + acw, crop_y + ach)); cropped_img.save(out_img / img_id)
                    preview_img = cropped_img.copy().convert("RGB"); draw = ImageDraw.Draw(preview_img)
                    id_to_name = {v: k for k, v in c["class_map"].items()}; color_palette = ["#00FF00", "#FF0000", "#00FFFF", "#FFFF00", "#FF00FF", "#0000FF", "#FFA500"]
                    for line in labels:
                        parts = line.split(); cid, cx, cy, nw, nh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        bw, bh = nw * acw, nh * ach; bx1, by1 = (cx * acw) - (bw / 2), (cy * ach) - (bh / 2)
                        class_name = id_to_name.get(cid, f"ID {cid}"); color = color_palette[cid % len(color_palette)]
                        draw.rectangle([bx1, by1, bx1 + bw, by1 + bh], outline=color, width=3); draw.text((bx1 + 2, max(0, by1 - 15)), class_name, fill=color)
                    preview_img.save(out_preview / img_id); img.close(); ok_count += 1; self.progress.emit(int((i + 1) / len(label_files) * 100))
                except Exception as ex: 
                    logger.debug(f"[Preprocess Thread] 이미지 처리 중 경고 발생 ({img_id}): {ex}"); continue
            
            logger.info(f"[Preprocess Thread] 데이터 전처리 완료. 성공: {ok_count}건")
            self.finished_ok.emit(ok_count)
        except Exception as e: 
            logger.error("[Preprocess Thread] 치명적 오류 발생", exc_info=True)
            self.error.emit(traceback.format_exc())
