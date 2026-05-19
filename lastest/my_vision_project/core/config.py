from ..utils import *
from ..utils.logger import logger

class ConfigDefaults:
    GLOBAL = {"model": "yolov8n.pt", "imgsz": "640"}
    
    TAB1 = {"auto_crop": True, "margin": 50, "mw": 1280, "mh": 960, "mx": 0, "my": 0, "clean": True, "exif": True, "class_map": "OK\nNG"}
    
    TAB2 = {
        "epochs": 400, "batch": 16, "workers": 8, "patience": 100, "folds": 5, "test_split": 0.2,
        "lcls": 0.5, "lbox": 7.5, "ldfl": 1.5, "tune_iterations": 30,
        "ah": 0.015, "as": 0.7, "av": 0.4, "adeg": 0.0, "atrans": 0.1, "ascale": 0.5,
        "ashear": 0.0, "afud": 0.0, "aflr": 0.5, "amos": 1.0, "amix": 0.0, "acp": 0.0
    }
    
    TAB3 = {
        "conf": 0.25, "iou": 0.45, "match_iou": 0.50, "max_det": 99, 
        "run_name": "check01", "agnostic": True, "save_rel": True
    }
    
    TAB4 = {
        "epochs": 10, "batch": 16, "run": "retrain_hard_01", "lcls": 0.5, "lbox": 7.5,
        "ah": 0.015, "as": 0.7, "av": 0.4, "afud": 0.0, "aflr": 0.5, "amos": 1.0, "amix": 0.0, "acp": 0.0,
        "eval_conf": 0.25, "eval_iou": 0.45, "eval_match": 0.50, "eval_max": 99, "eval_agnostic": True
    }
    
    TAB5 = {
        "method": "테두리 최단거리 (Edge)", "conf": 0.25, "iou": 0.45, "max_det": 300, 
        "agnostic": True, "knn_n": 2, "edge_thr": "60", "skip_ng": True, "drop_odd": False,
        "color1": "노란색 (Yellow)", "color2": "청록색 (Cyan)"
    }
    
    TAB6 = {"auto_thr": 0.75, "auto_nms": 0.3, "color": "흰색 (White)", "class_map": "OK\nNG"}

    @staticmethod
    def validate_and_fix(loaded_config, schema_template):
        """재귀적으로 타입을 검사하고 누락된 키나 잘못된 값을 복구합니다 (Improvement 5)"""
        validated = {}
        for key, default_val in schema_template.items():
            if key not in loaded_config:
                validated[key] = default_val
            elif isinstance(default_val, dict) and isinstance(loaded_config[key], dict):
                validated[key] = ConfigDefaults.validate_and_fix(loaded_config[key], default_val)
            else:
                try:
                    if isinstance(default_val, bool):
                        validated[key] = str(loaded_config[key]).lower() in ('true', '1', 't', 'y')
                    else:
                        validated[key] = type(default_val)(loaded_config[key])
                except (ValueError, TypeError):
                    validated[key] = default_val
        return validated

class ConfigManager:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir); self._workspace_path = None; self._config_dir = None
        logger.debug(f"ConfigManager 초기화됨. base_dir: {self.base_dir}")
        
    @property
    def config_dir(self):
        if self._config_dir is None: return self.base_dir / "workspace" / "configs"
        return self._config_dir

    def update_workspace_path(self, workspace_path):
        self._workspace_path = Path(workspace_path); self._config_dir = self._workspace_path / "configs"
        logger.info(f"Workspace 경로 업데이트됨: {self._workspace_path}")

    def save_config(self, config_data, config_name):
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if not config_name: 
                config_name = f"config_{timestamp}"
            else:
                config_name = Path(config_name).name
                
            file_path = self.config_dir / f"{config_name}.json"
            final_data = {"metadata": {"saved_at": timestamp, "version": "1.0", "name": config_name}, "config": config_data}
            
            with open(file_path, 'w', encoding='utf-8') as f: 
                json.dump(final_data, f, indent=4, ensure_ascii=False)
                
            logger.info(f"설정 저장 성공: {file_path}")
            return True, str(file_path)
        except Exception as e: 
            logger.error(f"설정 저장 실패: {e}", exc_info=True)
            return False, str(e)

    def load_config(self, file_path):
        try:
            logger.debug(f"설정 파일 불러오기 시도: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            
            raw_config = data.get("config", {})
            schema = {
                "global": ConfigDefaults.GLOBAL, "tab1": ConfigDefaults.TAB1, "tab2": ConfigDefaults.TAB2,
                "tab3": ConfigDefaults.TAB3, "tab4": ConfigDefaults.TAB4, "tab5": ConfigDefaults.TAB5, "tab6": ConfigDefaults.TAB6
            }
            # [개선 5] 불러온 데이터를 스키마에 맞춰 검증 및 자동 복원
            valid_config = ConfigDefaults.validate_and_fix(raw_config, schema)
            
            logger.info(f"설정 불러오기 성공 및 검증 완료: {file_path}")
            return True, valid_config
        except Exception as e: 
            logger.error(f"설정 불러오기 실패: {e}", exc_info=True)
            return False, str(e)

    def get_all_configs(self):
        return sorted(list(self.config_dir.glob("*.json")), key=os.path.getmtime, reverse=True)

class ConfigBuilder:
    @staticmethod
    def build(w):
        return {
            "global": {"proj_root": w.w_proj_root.get_path(), "base_ds": w.w_base_ds.get_path(), "proc_ds": w.w_proc_ds.get_path(), "work_ds": w.w_work_ds.get_path(), "webhook_url": w.webhook_url, "noti_flags": w.get_noti_flags(), "model": w.g_model.currentText(), "imgsz": w.g_imgsz.currentText()},
            "tab1": {"auto_crop": w.t1_auto_crop.isChecked(), "margin": w.t1_margin.value(), "mx": w.t1_mx.value(), "my": w.t1_my.value(), "mw": w.t1_mw.value(), "mh": w.t1_mh.value(), "class_map": w.t1_class_map.toPlainText(), "clean": w.t1_clean.isChecked(), "exif": w.t1_exif.isChecked()},
            "tab2": {"epochs": w.t2_epochs.value(), "batch": w.t2_batch.value(), "workers": w.t2_workers.value(), "patience": w.t2_patience.value(), "folds": w.t2_folds.value(), "test_split": w.t2_test_split.value(), "lcls": w.t2_lcls.value(), "lbox": w.t2_lbox.value(), "ldfl": w.t2_ldfl.value(), "tune_iterations": w.t2_tune_iterations.value(), "ah": w.t2_ah.value(), "as": w.t2_as.value(), "av": w.t2_av.value(), "adeg": w.t2_adeg.value(), "atrans": w.t2_atrans.value(), "ascale": w.t2_ascale.value(), "ashear": w.t2_ashear.value(), "afud": w.t2_afud.value(), "aflr": w.t2_aflr.value(), "amos": w.t2_amos.value(), "amix": w.t2_amix.value(), "acp": w.t2_acp.value()},
            "tab3": {"conf": w.t3_conf.value(), "iou": w.t3_iou.value(), "match_iou": w.t3_match_iou.value(), "max_det": w.t3_max_det.value(), "run_name": w.t3_run_name.text(), "agnostic": w.t3_agnostic.isChecked(), "save_rel": w.t3_save_rel.isChecked()},
            "tab4": {"epochs": w.t4_epochs.value(), "batch": w.t4_batch.value(), "run": w.t4_run.text(), "lcls": w.t4_lcls.value(), "lbox": w.t4_lbox.value(), "ah": w.t4_ah.value(), "as": w.t4_as.value(), "av": w.t4_av.value(), "afud": w.t4_afud.value(), "aflr": w.t4_aflr.value(), "amos": w.t4_amos.value(), "amix": w.t4_amix.value(), "acp": w.t4_acp.value(), "eval_conf": w.t4_conf.value(), "eval_iou": w.t4_iou.value(), "eval_match": w.t4_match_iou.value(), "eval_max": w.t4_max_det.value(), "eval_agnostic": w.t4_agnostic.isChecked()},
            "tab5": {"method": w.t5_method.currentText(), "conf": w.t5_conf.value(), "iou": w.t5_iou.value(), "max_det": w.t5_max_det.value(), "agnostic": w.t5_agnostic.isChecked(), "knn_n": w.t5_knn_n.value(), "edge_thr": w.t5_edge_thr.text(), "skip_ng": w.t5_skip_ng.isChecked(), "drop_odd": w.t5_drop_odd.isChecked(), "color1": w.t5_color1.currentText(), "color2": w.t5_color2.currentText()},
            "tab6": {"class_map": w.t6_class_map.toPlainText(), "color": w.t6_color_combo.currentText(), "auto_thr": w.t6_auto_thr.value(), "auto_nms": w.t6_auto_nms.value()}
        }
