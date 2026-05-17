import sys, os, json, shutil, math, time, platform, multiprocessing, signal, re, gc, traceback, sqlite3, csv, html, subprocess
import queue as qlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta
import zipfile, cv2, numpy as np, pandas as pd, psutil, torch, yaml
import threading

from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal, QTimer, QRectF, QSettings, QDate
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QFormLayout, 
    QScrollArea, QGridLayout, QSplitter, QDialog, QInputDialog, QGraphicsView, QGraphicsScene, 
    QGraphicsRectItem, QGraphicsPixmapItem, QListWidget, QListWidgetItem, QGraphicsLineItem,
    QDateEdit)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPainter, QPen

if hasattr(Qt, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

# ==========================================
# [개선] 통합 기본값 관리 (Config Defaults)
# ==========================================
class ConfigDefaults:
    GLOBAL = {"model": "yolov8n.pt", "imgsz": "640"}
    
    TAB1 = {"auto_crop": True, "margin": 50, "mw": 1280, "mh": 960, "clean": True, "exif": True}
    
    TAB2 = {
        "epochs": 400, "batch": 16, "workers": 8, "patience": 100, "seed": 42, "folds": 5, "test_split": 0.2,
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
    
    TAB6 = {"auto_thr": 0.75, "auto_nms": 0.3, "color": "흰색 (White)"}

# ==========================================
# 로깅(Logging) 설정
# ==========================================
class EmojiFormatter(logging.Formatter):
    LEVEL_EMOJIS = {
        logging.DEBUG: "🔍 [DEBUG]",
        logging.INFO: "ℹ️ [INFO]",
        logging.WARNING: "⚠️ [WARN]",
        logging.ERROR: "❌ [ERROR]",
        logging.CRITICAL: "🔴 [CRITICAL]"
    }

    def format(self, record):
        original_levelname = record.levelname
        record.levelname = self.LEVEL_EMOJIS.get(record.levelno, original_levelname)
        result = super().format(record)
        record.levelname = original_levelname
        return result

def setup_logger():
    logger = logging.getLogger("YOLO_Pipeline")
    if logger.hasHandlers(): return logger
    logger.setLevel(logging.DEBUG)
    
    base_dir = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    file_handler = RotatingFileHandler(log_dir / "app.log", maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('[%(levelname)s] %(asctime)s - [PID:%(process)d|%(threadName)s] - %(funcName)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = EmojiFormatter('%(levelname)s %(asctime)s - [%(threadName)s] - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger

logger = setup_logger()

def clear_vram():
    logger.debug("VRAM 정리(Garbage Collection & Cuda empty cache) 시작")
    gc.collect()
    try:
        if torch.cuda.is_available(): torch.cuda.empty_cache(); logger.debug("CUDA empty_cache 완료")
    except Exception as e: 
        logger.error(f"VRAM 정리 중 오류 발생: {e}", exc_info=True)


class GracefulStopHandler:
    def __init__(self, stop_event, logger, queue=None, default_result=None):
        self.stop_event = stop_event
        self.logger = logger
        self.queue = queue
        self.default_result = default_result or {"success": False, "error": "사용자에 의해 취소되었습니다."}

    def should_stop(self, context="", put_queue=True):
        if self.stop_event and self.stop_event.is_set():
            self.logger.info(f"🛑 사용자 중단 신호 감지 [{context}]")
            if put_queue and self.queue:
                self.default_result["error"] = f"작업이 취소되었습니다. ({context})"
                self.queue.put(self.default_result)
            return True
        return False

    def check_every_n_iterations(self, iteration, interval=10, context="", put_queue=True):
        if iteration % interval == 0:
            return self.should_stop(context, put_queue)
        return False

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
            logger.info(f"설정 불러오기 성공: {file_path}")
            return True, data.get("config", {})
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
            "tab2": {"epochs": w.t2_epochs.value(), "batch": w.t2_batch.value(), "workers": w.t2_workers.value(), "patience": w.t2_patience.value(), "seed": w.t2_seed.value(), "folds": w.t2_folds.value(), "test_split": w.t2_test_split.value(), "lcls": w.t2_lcls.value(), "lbox": w.t2_lbox.value(), "ldfl": w.t2_ldfl.value(), "ah": w.t2_ah.value(), "as": w.t2_as.value(), "av": w.t2_av.value(), "adeg": w.t2_adeg.value(), "atrans": w.t2_atrans.value(), "ascale": w.t2_ascale.value(), "ashear": w.t2_ashear.value(), "afud": w.t2_afud.value(), "aflr": w.t2_aflr.value(), "amos": w.t2_amos.value(), "amix": w.t2_amix.value(), "acp": w.t2_acp.value()},
            "tab3": {"conf": w.t3_conf.value(), "iou": w.t3_iou.value(), "match_iou": w.t3_match_iou.value(), "max_det": w.t3_max_det.value(), "run_name": w.t3_run_name.text(), "agnostic": w.t3_agnostic.isChecked(), "save_rel": w.t3_save_rel.isChecked()},
            "tab4": {"epochs": w.t4_epochs.value(), "batch": w.t4_batch.value(), "run": w.t4_run.text(), "lcls": w.t4_lcls.value(), "lbox": w.t4_lbox.value(), "ah": w.t4_ah.value(), "as": w.t4_as.value(), "av": w.t4_av.value(), "afud": w.t4_afud.value(), "aflr": w.t4_aflr.value(), "amos": w.t4_amos.value(), "amix": w.t4_amix.value(), "acp": w.t4_acp.value(), "eval_conf": w.t4_conf.value(), "eval_iou": w.t4_iou.value(), "eval_match": w.t4_match_iou.value(), "eval_max": w.t4_max_det.value(), "eval_agnostic": w.t4_agnostic.isChecked()},
            "tab5": {"method": w.t5_method.currentText(), "conf": w.t5_conf.value(), "iou": w.t5_iou.value(), "max_det": w.t5_max_det.value(), "agnostic": w.t5_agnostic.isChecked(), "knn_n": w.t5_knn_n.value(), "edge_thr": w.t5_edge_thr.text(), "skip_ng": w.t5_skip_ng.isChecked(), "drop_odd": w.t5_drop_odd.isChecked(), "color1": w.t5_color1.currentText(), "color2": w.t5_color2.currentText()},
            "tab6": {"class_map": w.t6_class_map.toPlainText(), "color": w.t6_color_combo.currentText(), "auto_thr": w.t6_auto_thr.value(), "auto_nms": w.t6_auto_nms.value()}
        }

class LogDatabase:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        logger.debug(f"LogDatabase 초기화. 경로: {self.db_path}")

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_db(self):
        if not self.db_path.parent.exists(): 
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"DB 상위 폴더 생성됨: {self.db_path.parent}")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS training_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, task_type TEXT, model_name TEXT, epochs INTEGER, batch_size INTEGER, best_map REAL, save_dir TEXT, config_json TEXT)')
            cursor.execute('CREATE TABLE IF NOT EXISTS evaluation_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, task_type TEXT, model_name TEXT, total_imgs INTEGER, wrong_count INTEGER, accuracy REAL, wrong_images TEXT, config_json TEXT)')
            conn.commit()

    def insert_log(self, task_type, model_name, epochs, batch, best_map, save_dir, config_data):
        self._ensure_db()
        try:
            with self._get_connection() as conn:
                conn.execute('INSERT INTO training_logs (timestamp, task_type, model_name, epochs, batch_size, best_map, save_dir, config_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), task_type, model_name, epochs, batch, best_map, save_dir, json.dumps(config_data, ensure_ascii=False)))
                conn.commit()
            logger.info(f"학습 로그 DB 삽입 완료. Task: {task_type}, Best mAP: {best_map:.4f}")
        except Exception as e: logger.error(f"학습 로그 삽입 중 오류: {e}", exc_info=True)

    def insert_eval_log(self, task_type, model_name, total, wrong, accuracy, wrong_imgs_list, config_data):
        self._ensure_db()
        try:
            with self._get_connection() as conn:
                conn.execute('INSERT INTO evaluation_logs (timestamp, task_type, model_name, total_imgs, wrong_count, accuracy, wrong_images, config_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), task_type, model_name, total, wrong, accuracy, json.dumps([Path(p).name for p in wrong_imgs_list], ensure_ascii=False), json.dumps(config_data, ensure_ascii=False)))
                conn.commit()
            logger.info(f"평가 로그 DB 삽입 완료. Task: {task_type}, Acc: {accuracy:.2f}%")
        except Exception as e: logger.error(f"평가 로그 삽입 중 오류: {e}", exc_info=True)

    def delete_logs(self, table_type, ids):
        if not ids: return
        self._ensure_db()
        table_name = 'training_logs' if table_type == 'train' else 'evaluation_logs'
        placeholders = ','.join('?' for _ in ids)
        try:
            with self._get_connection() as conn:
                conn.execute(f'DELETE FROM {table_name} WHERE id IN ({placeholders})', ids)
                conn.commit()
            logger.info(f"DB 레코드 삭제 완료. Table: {table_name}, IDs: {ids}")
        except Exception as e: logger.error(f"DB 레코드 삭제 중 오류: {e}", exc_info=True)

    def fetch_logs(self, table_type, search_kw="", date_from=None, date_to=None, filter_ng=False, offset=0, limit=100, sort_col="id", sort_order="DESC"):
        self._ensure_db()
        table_name = 'training_logs' if table_type == 'train' else 'evaluation_logs'
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        params = []
        if search_kw:
            query += " AND (task_type LIKE ? OR model_name LIKE ?)"
            params.extend([f"%{search_kw}%", f"%{search_kw}%"])
        if date_from and date_to:
            query += " AND timestamp BETWEEN ? AND ?"
            params.extend([date_from + " 00:00:00", date_to + " 23:59:59"])
        if table_type == 'eval' and filter_ng:
            query += " AND wrong_count > 0"
            
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        try:
            with self._get_connection() as conn:
                total = conn.execute(count_query, params).fetchone()[0]
                if limit > 0:
                    query += f" ORDER BY {sort_col} {sort_order} LIMIT ? OFFSET ?"
                    params.extend([limit, offset])
                else:
                    query += f" ORDER BY {sort_col} {sort_order}"
                rows = conn.execute(query, params).fetchall()
            logger.debug(f"DB 로그 조회 완료. Table: {table_name}, Count: {len(rows)}/{total}")
            return rows, total
        except Exception as e:
            logger.error(f"DB 로그 조회 중 오류: {e}", exc_info=True)
            return [], 0

def open_folder(path):
    p = Path(path)
    if not p.exists(): logger.warning(f"폴더를 열 수 없습니다. 존재하지 않음: {path}"); return
    target_dir = str(p.parent) if p.is_file() else str(p)
    logger.debug(f"폴더 열기 실행: {target_dir}")
    if platform.system() == "Windows": os.startfile(target_dir)
    elif platform.system() == "Darwin": subprocess.Popen(["open", target_dir])
    else: subprocess.Popen(["xdg-open", target_dir])

def send_discord_webhook(webhook_url, title, description, color=0x3498db, fields=None, retry_count=2, sync=False):
    if not webhook_url or not webhook_url.startswith("http"): 
        return False

    def _send_task():
        import requests
        import time
        from datetime import datetime
        from requests.exceptions import Timeout, ConnectionError, RequestException

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        if fields:
            embed["fields"] = fields

        payload = {"embeds": [embed]}

        for attempt in range(retry_count):
            try:
                logger.debug(f"Discord 임베드 웹훅 전송 시도. (시도 {attempt+1}/{retry_count})")
                response = requests.post(
                    webhook_url, 
                    json=payload, 
                    timeout=5
                )
                response.raise_for_status()
                logger.debug("웹훅 전송 성공")
                return
                
            except Timeout:
                logger.warning(f"웹훅 타임아웃 (시도 {attempt+1}/{retry_count})")
            except ConnectionError as e:
                logger.error(f"네트워크 연결 실패: {e}")
            except RequestException as e:
                logger.error(f"웹훅 HTTP 에러 (Rate Limit 등): {e}")
            except Exception as e:
                logger.error(f"웹훅 전송 중 알 수 없는 오류: {e}")
            
            if attempt < retry_count - 1:
                time.sleep(2)

        logger.error("디스코드 웹훅 최종 전송 실패")
        
    if sync:
        _send_task()  # 스레드를 쓰지 않고 그 자리에서 전송을 끝낼 때까지 대기
    else:
        import threading
        threading.Thread(target=_send_task, daemon=True).start()  # 평소처럼 백그라운드 전송
    return True

def create_heartbeat_callback(webhook_url, total_epochs, interval):
    if interval <= 0: return lambda trainer: None
    
    def on_train_epoch_end(trainer):
        current_epoch = trainer.epoch + 1
        if current_epoch % interval == 0:
            logger.debug(f"웹훅 Heartbeat 발생: Epoch {current_epoch}/{total_epochs}")
            
            # tloss 계산 중 발생할 수 있는 Ultralytics 내부 에러 방어
            loss_val = "N/A"
            try:
                if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                    loss_val = f"`{trainer.tloss.sum().item():.4f}`"
            except Exception as e:
                logger.warning(f"웹훅: Loss 값 추출 실패 ({e})")

            fields = [
                {"name": "진척도 (Epochs)", "value": f"{current_epoch} / {total_epochs}", "inline": True},
                {"name": "Total Loss", "value": loss_val, "inline": True}
            ]
            
            send_discord_webhook(
                webhook_url=webhook_url,
                title="💓 [학습 진행 상황]",
                description="모델 학습이 정상적으로 진행 중입니다.",
                color=0x3498db,
                fields=fields
            )
    return on_train_epoch_end

def create_stop_callback(stop_event):
    def on_train_epoch_end(trainer):
        if stop_event is not None and stop_event.is_set():
            logger.info("🛑 사용자의 중지 요청 감지. 이번 Epoch를 끝으로 학습을 안전하게 조기 종료합니다.")
            trainer.stop = True
    return on_train_epoch_end

class WebhookSettingsDialog(QDialog):
    def __init__(self, current_flags, current_url, parent=None):
        super().__init__(parent); self.setWindowTitle("디스코드 웹훅 상세 설정"); self.resize(400, 480)
        layout = QVBoxLayout(self)
        grp_url = QGroupBox("디스코드 웹훅 연동"); l_url = QVBoxLayout(grp_url)
        self.txt_url = QLineEdit(current_url); self.txt_url.setPlaceholderText("https://discord.com/api/webhooks/...")
        self.txt_url.setEchoMode(QLineEdit.Password); l_url.addWidget(self.txt_url); layout.addWidget(grp_url)
        grp_sys = QGroupBox("시스템 알림"); l_sys = QVBoxLayout(grp_sys)
        self.chk_error = QCheckBox("🚨 에러 및 비정상 종료 (권장)"); self.chk_early_stop = QCheckBox("🛑 조기 종료 (Early Stop) 발동")
        l_sys.addWidget(self.chk_error); l_sys.addWidget(self.chk_early_stop); layout.addWidget(grp_sys)
        grp_prog = QGroupBox("진행 상황 알림"); l_prog = QVBoxLayout(grp_prog)
        self.chk_start = QCheckBox("▶️ 작업 시작 (학습, 튜닝 등)"); self.chk_fold = QCheckBox("📍 K-Fold 각 Fold 완료 시"); self.chk_tune = QCheckBox("🤖 Auto ML 세대별 탐색 진행 상황")
        h_epoch = QHBoxLayout(); self.chk_epoch = QCheckBox("💓 학습 에포크 진행 상황"); self.cmb_epoch = QComboBox()
        self.cmb_epoch.addItems(["10 Epoch 마다", "50 Epoch 마다", "100 Epoch 마다", "500 Epoch 마다"])
        h_epoch.addWidget(self.chk_epoch); h_epoch.addWidget(self.cmb_epoch)
        l_prog.addWidget(self.chk_start); l_prog.addWidget(self.chk_fold); l_prog.addWidget(self.chk_tune); l_prog.addLayout(h_epoch)
        layout.addWidget(grp_prog)
        grp_task = QGroupBox("완료 알림"); l_task = QVBoxLayout(grp_task)
        self.chk_task = QCheckBox("✅ 주요 작업 완료 시"); l_task.addWidget(self.chk_task); layout.addWidget(grp_task)

        self.chk_error.setChecked(current_flags.get("error", True)); self.chk_early_stop.setChecked(current_flags.get("early_stop", True))
        self.chk_start.setChecked(current_flags.get("start", True)); self.chk_fold.setChecked(current_flags.get("fold", True))
        self.chk_tune.setChecked(current_flags.get("tune", True)); self.chk_epoch.setChecked(current_flags.get("epoch", True))
        self.chk_task.setChecked(current_flags.get("task", True))
        interval = current_flags.get("epoch_interval", 100)
        idx = ["10", "50", "100", "500"].index(str(interval)) if str(interval) in ["10", "50", "100", "500"] else 2
        self.cmb_epoch.setCurrentIndex(idx)
        self.chk_epoch.stateChanged.connect(lambda state: self.cmb_epoch.setEnabled(state == Qt.Checked)); self.cmb_epoch.setEnabled(self.chk_epoch.isChecked())
        btn_layout = QHBoxLayout(); btn_ok = QPushButton("저장"); btn_ok.clicked.connect(self.accept); btn_cancel = QPushButton("취소"); btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok); btn_layout.addWidget(btn_cancel); layout.addLayout(btn_layout)

    def get_url(self): return self.txt_url.text().strip()
    def get_flags(self): return {"error": self.chk_error.isChecked(), "early_stop": self.chk_early_stop.isChecked(), "start": self.chk_start.isChecked(), "fold": self.chk_fold.isChecked(), "tune": self.chk_tune.isChecked(), "epoch": self.chk_epoch.isChecked(), "epoch_interval": {0: 10, 1: 50, 2: 100, 3: 500}[self.cmb_epoch.currentIndex()], "task": self.chk_task.isChecked()}

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
            model.add_callback("on_train_epoch_end", create_stop_callback(args["stop_event"]))

        res = model.train(
            data=str(args["data_yaml"]), epochs=args["adaptive_epochs"], patience=args["tune_patience"], 
            batch=args["batch"], workers=args["workers"], project=str(args["tune_base"]), 
            name=f"gen_{args['gen']+1}", seed=42, verbose=False, **args["current_params"]
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
    
    stop_handler = GracefulStopHandler(args.get("stop_event"), worker_logger, queue, result)
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
            
        tr, vl = train_test_split(paired, test_size=0.2, random_state=42)
        tr_txt, vl_txt = tune_base / "train.txt", tune_base / "val.txt"
        tr_txt.write_text("\n".join(str(Path(p[0]).resolve()) for p in tr))
        vl_txt.write_text("\n".join(str(Path(p[0]).resolve()) for p in vl))
        data_yaml = tune_base / "tune_data.yaml"
        data_yaml.write_text(f"train: {tr_txt.resolve()}\nval: {vl_txt.resolve()}\nnc: {len(args['class_names'])}\nnames: {args['class_names']}\n")
        
        search_history = []
        
        def objective(trial):
            if stop_handler.should_stop("Optuna 탐색 단계(Objective)", put_queue=False):
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
                "stop_event": args.get("stop_event")
            }
            
            run_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=_single_tune_run, args=(run_args, run_queue))
            p.start()
            p.join() 
            
            if not run_queue.empty():
                run_res = run_queue.get()
                if not run_res["success"]: raise Exception(f"세대 {gen+1} 학습 중 오류: {run_res.get('error')}")
                actual_epochs, mAP50, mAP50_95, fitness = run_res["actual_epochs"], run_res["mAP50"], run_res["mAP50_95"], run_res["fitness"]
            else:
                raise Exception(f"세대 {gen+1} 워커 프로세스가 비정상 종료되었습니다.")
            
            if actual_epochs < adaptive_epochs and args.get("webhook_url") and args.get("noti_flags", {}).get("early_stop", False):
                worker_logger.info(f"세대 {gen+1} 조기 종료 감지됨 (Epoch: {actual_epochs})")
                send_discord_webhook(
                    webhook_url=args["webhook_url"],
                    title="🛑 [조기 종료 발동]",
                    description=f"Auto ML {gen+1}세대 - **{actual_epochs} Epoch**에서 학습이 조기 종료되었습니다.",
                    color=0xe74c3c
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
        except Exception as e:
            if stop_handler.should_stop("최적화 진행 중 예외 발생", put_queue=False):
                worker_logger.info("탐색이 안전하게 조기 종료되었습니다. 지금까지의 결과를 반환합니다.")
            else:
                raise e

        if not search_history:
            result["error"] = "탐색 결과가 없습니다."
            queue.put(result)
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
                fields=fields
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
    
    stop_handler = GracefulStopHandler(args.get("stop_event"), worker_logger, queue, result)
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
            model.add_callback("on_train_epoch_end", create_stop_callback(args["stop_event"]))

        res = model.train(
            data=str(args["data_yaml"]), epochs=args["epochs"], patience=args["patience"], imgsz=args["imgsz"], 
            batch=args["batch"], workers=args["workers"], project=str(args["runs_dir"]), name=args["run_name"], 
            seed=args["seed"], deterministic=args["deterministic"], verbose=True, **args["aug"], 
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
    stop_handler = GracefulStopHandler(args.get("stop_event"), worker_logger, queue, result)
    worker_logger.info(f"[K-Fold Worker] 학습 시작. Folds: {args['num_folds']}, Model: {args['model_name']}, Epochs: {args['epochs']}")
    
    try:
        processed_dir, workspace_dir = Path(args["processed_dir"]), Path(args["workspace_dir"])
        kfold_base = workspace_dir / "kfold"; runs_dir = workspace_dir / "runs" / "kfold_train"
        img_map = {f.stem: f for f in sorted((processed_dir / "images").glob("*.jpg"))}
        lbl_map = {f.stem: f for f in sorted((processed_dir / "labels").glob("*.txt"))}
        paired = [(str(img_map[n]), str(lbl_map[n])) for n in img_map if n in lbl_map]
        worker_logger.debug(f"[K-Fold Worker] 매칭된 데이터 수: {len(paired)}개")
        if len(paired) < 5: result["error"] = "데이터 부족"; worker_logger.error("데이터 부족으로 종료"); queue.put(result); return
        train_val, test_files = train_test_split(paired, test_size=args["test_split"], random_state=args["random_seed"])
        if kfold_base.exists(): shutil.rmtree(kfold_base); worker_logger.debug("기존 kfold 폴더 초기화 완료")
        kfold_base.mkdir(parents=True, exist_ok=True)
        test_img_dir = kfold_base / "images" / "test"; test_lbl_dir = kfold_base / "labels" / "test"; test_img_dir.mkdir(parents=True, exist_ok=True); test_lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl in test_files: shutil.copy(img, test_img_dir); shutil.copy(lbl, test_lbl_dir)
        fold_metrics, fold_save_dirs = [], {}
        splits = [(train_test_split(train_val, test_size=args["test_split"], random_state=args["random_seed"]))] if args["num_folds"] == 1 else list(KFold(n_splits=args["num_folds"], shuffle=True, random_state=args["random_seed"]).split(train_val))
        
        for fold, split_data in enumerate(splits):
            if stop_handler.should_stop(f"Fold {fold+1} 시작 전", put_queue=False):
                if not fold_metrics:
                    result["error"] = "사용자에 의해 학습이 취소되었습니다."
                    queue.put(result)
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
                "seed": args["random_seed"], "deterministic": args["deterministic"], "aug": args["aug"], "loss": args["loss"],
                "webhook_url": args.get("webhook_url"), "noti_flags": args.get("noti_flags", {}), "stop_event": args.get("stop_event")
            }
            
            run_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=_single_fold_run, args=(run_args, run_queue))
            p.start()
            p.join()
            
            if not run_queue.empty():
                run_res = run_queue.get()
                if not run_res["success"]: raise Exception(f"Fold {fold_num} 학습 중 오류: {run_res.get('error')}")
                actual_epochs, res_dict, save_dir = run_res["actual_epochs"], run_res["results_dict"], run_res["save_dir"]
            else:
                raise Exception(f"Fold {fold_num} 워커 프로세스가 비정상 종료되었습니다.")
            
            if actual_epochs < args["epochs"]:
                worker_logger.info(f"Fold {fold_num} 조기 종료 감지. (Epoch: {actual_epochs})")
                if args.get("webhook_url") and args.get("noti_flags", {}).get("early_stop", False):
                    send_discord_webhook(
                        webhook_url=args["webhook_url"],
                        title="🛑 [조기 종료 발동]",
                        description=f"Fold {fold_num} - **{actual_epochs} Epoch**에서 학습이 조기 종료되었습니다.",
                        color=0xe74c3c
                    )

            fold_metrics.append(res_dict); fold_save_dirs[fold_num] = save_dir
            
            if args.get("webhook_url") and args.get("noti_flags", {}).get("fold", False):
                mAP = res_dict.get('metrics/mAP50-95(B)', 0)
                send_discord_webhook(
                    webhook_url=args["webhook_url"],
                    title=f"📍 [K-Fold] Fold {fold_num} 완료",
                    description=f"검증 mAP50-95: **{mAP:.4f}**",
                    color=0x2ecc71
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
        files = [f for f in Path(p["rt_hard_dir"]).iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
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
            model.add_callback("on_train_epoch_end", create_stop_callback(p["stop_event"]))

        res = model.train(data=str(yaml_rt), epochs=p["rt_epochs"], imgsz=p["imgsz"], batch=p["rt_batch"], project=str(runs_dir), name=p["rt_run_name"], exist_ok=True, hsv_h=p["rt_h"], hsv_s=p["rt_s"], hsv_v=p["rt_v"], flipud=p["rt_flipud"], fliplr=p["rt_fliplr"], mosaic=p["rt_mosaic"], mixup=p["rt_mix"], copy_paste=p["rt_cp"], cls=p["rt_cls"], box=p["rt_box"], verbose=True)
        
        actual_epochs = len(pd.read_csv(Path(res.save_dir) / "results.csv")) if (Path(res.save_dir) / "results.csv").exists() else p["rt_epochs"]
        if actual_epochs < p["rt_epochs"] and p.get("webhook_url") and p.get("noti_flags", {}).get("early_stop", False):
            worker_logger.info(f"[Retrain Worker] 조기 종료됨 (Epoch: {actual_epochs})")
            send_discord_webhook(
                webhook_url=p["webhook_url"],
                title="🛑 [조기 종료 발동]",
                description=f"재학습이 **{actual_epochs} Epoch**에서 조기 종료되었습니다.",
                color=0xe74c3c
            )

        trained_model_path = Path(res.save_dir) / "weights" / "best.pt"; del model
        result["success"] = True; result["msg"] = f"✅ 재학습 완료!\n저장: {res.save_dir}"; result["save_dir"] = str(res.save_dir); result["model_path"] = str(trained_model_path)
        worker_logger.info("[Retrain Worker] 하드 재학습 모두 완료됨")
    except Exception: result["error"] = traceback.format_exc(); worker_logger.error(f"[Retrain Worker] Exception: {result['error']}")
    finally: clear_vram(); queue.put(result)

class ProcessMonitorThread(QThread):
    finished_ok = pyqtSignal(dict); error = pyqtSignal(str)
    def __init__(self, queue, process): super().__init__(); self.queue = queue; self.process = process
    def run(self):
        logger.debug(f"[Monitor Thread] 모니터링 시작 (PID: {self.process.pid})")
        while self.process.is_alive():
            try: self.finished_ok.emit(self.queue.get(timeout=0.5)); logger.debug("[Monitor Thread] Worker 결과 수신 성공"); return
            except qlib.Empty: continue
        try: self.finished_ok.emit(self.queue.get(timeout=1.0))
        except qlib.Empty:
            if self.process.exitcode != 0: 
                msg = f"프로세스 비정상 종료 (Exit Code: {self.process.exitcode})"
                logger.error(msg)
                self.error.emit(msg)

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
                        img_path = next((image_dir / lf.with_suffix(ext).name for ext in [".jpg", ".jpeg", ".png"] if (image_dir / lf.with_suffix(ext).name).exists()), None)
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
                    temp = next((image_dir / lf.with_suffix(ext).name for ext in [".jpg", ".jpeg", ".png"] if (image_dir / lf.with_suffix(ext).name).exists()), None)
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

class EvalThread(QThread):
    finished_ok = pyqtSignal(pd.DataFrame, list, list, dict); error = pyqtSignal(str)
    def __init__(self, config): super().__init__(); self.config = config
    def run(self):
        logger.info(f"[Eval Thread] 평가 시작. 대상 모델: {self.config['eval_model_path']}")
        try:
            from ultralytics import YOLO
            c = self.config; eval_project_dir = Path(c["workspace_dir"]) / "runs" / "eval"
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
                img_name = Path(r.path).name; current_img_path = str(eval_project_dir / c["eval_run_name"] / img_name); all_imgs.append(current_img_path)
                pred_boxes = [{"class": int(cls.item()), "box": b.tolist()} for cls, b in zip(r.boxes.cls, r.boxes.xywhn)] if len(r.boxes) > 0 else []
                gt_boxes = []; txt = Path(c["gt_labels_path"]) / Path(img_name).with_suffix(".txt").name
                if txt.exists():
                    for line in txt.read_text().splitlines():
                        parts = line.strip().split(); 
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
            logger.info(f"[Eval Thread] 평가 완료. F1-Score: {f1_score*100:.2f}%, 오답: {wrong}건")
            del model; clear_vram()
            self.finished_ok.emit(pd.DataFrame(rows).sort_values("상태"), wrong_imgs, all_imgs, stats)
        except Exception as e: 
            logger.error("[Eval Thread] 에러 발생", exc_info=True)
            self.error.emit(traceback.format_exc())

class Tab4FinalEvalThread(QThread):
    finished_ok = pyqtSignal(pd.DataFrame, list, list, dict, str); error = pyqtSignal(str)
    def __init__(self, config): super().__init__(); self.config = config
    def run(self):
        logger.info("[Tab4FinalEval Thread] 재학습 모델 최종 평가 시작")
        try:
            from ultralytics import YOLO
            c = self.config; model = YOLO(str(c["retrained_model"]))
            val_metrics = model.val(data=str(c["yaml_path"]), conf=c["eval_conf"], iou=c["eval_iou"], max_det=c["max_det"], project=str(Path(c["workspace_dir"]) / "runs" / "eval"), name=c["eval_run_name"] + "_val")
            pr_curve = str(Path(val_metrics.save_dir) / "PR_curve.png"); eval_config = c.copy(); eval_config["eval_model_path"] = c["retrained_model"]; eval_config["eval_run_name"] = c["eval_run_name"] + "_final_eval"; eval_config["save_relabel"] = False
            del model; clear_vram(); eval_thread = EvalThread(eval_config)
            eval_thread.finished_ok.connect(lambda df, imgs, a_imgs, st: self.finished_ok.emit(df, imgs, a_imgs, st, pr_curve))
            eval_thread.error.connect(self.error.emit); eval_thread.run()
        except Exception as e: 
            logger.error("[Tab4FinalEval Thread] 에러 발생", exc_info=True)
            self.error.emit(traceback.format_exc())

class MeasureThread(QThread):
    progress = pyqtSignal(int); log_msg = pyqtSignal(str); finished_ok = pyqtSignal(pd.DataFrame, pd.DataFrame, pd.DataFrame, list); error = pyqtSignal(str)
    COLOR_MAP = {"노란색 (Yellow)": (0, 255, 255), "초록색 (Green)": (0, 255, 0), "빨간색 (Red)": (0, 0, 255), "파란색 (Blue)": (255, 0, 0), "청록색 (Cyan)": (255, 255, 0), "자주색 (Magenta)": (255, 0, 255), "흰색 (White)": (255, 255, 255)}
    def __init__(self, config): super().__init__(); self.config = config
    def run(self):
        logger.info(f"[Measure Thread] 거리 측정 시작 (Method: {self.config['measure_method']})")
        try:
            from ultralytics import YOLO
            c = self.config; dist_source = Path(c["dist_source"])
            is_edge, is_knn = c["measure_method"] == "테두리 최단거리 (Edge)", c["measure_method"] == "가장 가까운 N개 이웃 (방향 무관)"
            c1 = self.COLOR_MAP.get(c.get("color1", "노란색 (Yellow)"), (0, 255, 255)); c2 = self.COLOR_MAP.get(c.get("color2", "청록색 (Cyan)"), (255, 255, 0))
            dist_run_name = "ok_edge_distance" if is_edge else "ok_knn_distance" if is_knn else "ok_euclidean_distance"
            dist_col_name = "수평/수직 테두리 거리(px)" if is_edge else "최단 중심점 거리_N개(px)" if is_knn else "중심점 간 유클리드 거리(px)"
            base_dir = Path(c["workspace_dir"]) / "runs" / "measure"; base_dir.mkdir(parents=True, exist_ok=True); folder_idx = 1
            while (base_dir / f"{dist_run_name}_{folder_idx:02d}").exists(): folder_idx += 1
            final_save_dir = base_dir / f"{dist_run_name}_{folder_idx:02d}"; final_save_dir.mkdir()
            model = YOLO(str(c["dist_model_path"]))
            img_files = [f for f in dist_source.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]; total_imgs = len(img_files)
            results = model.predict(source=str(dist_source), save=False, conf=c["dist_conf"], iou=c["dist_iou"], max_det=c["dist_max_det"], agnostic_nms=c["dist_agnostic"], stream=True)
            image_pairs = []; processed = 0
            for r in results:
                processed += 1; self.progress.emit(int(processed / max(total_imgs, 1) * 100))
                if processed % 50 == 0: logger.debug(f"[Measure Thread] 처리 중: {processed}/{total_imgs}")
                img_name = Path(r.path).name; result_img_path = final_save_dir / img_name
                detected = [r.names[int(cls_id)].lower() for cls_id in r.boxes.cls]
                if c["skip_ng"] and "ng" in detected: continue
                num_objects = len(r.boxes)
                if c["drop_odd_lowest"] and num_objects % 2 != 0 and num_objects > 0:
                    r = r[[i for i in range(num_objects) if i != int(r.boxes.conf.argmin().item())]]
                    num_objects = len(r.boxes)
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
                                if oy > 0 and b2[0] >= b1[2] and (b2[0] - b1[2]) < min_x:
                                    min_x = r_dist = b2[0] - b1[2]; y_cen = int((max(b1[1], b2[1]) + min(b1[3], b2[3])) / 2); r_pts = ((int(b1[2]), y_cen), (int(b2[0]), y_cen))
                                if ox > 0 and b2[1] >= b1[3] and (b2[1] - b1[3]) < min_y:
                                    min_y = b_dist = b2[1] - b1[3]; x_cen = int((max(b1[0], b2[0]) + min(b1[2], b2[2])) / 2); b_pts = ((x_cen, int(b1[3])), (x_cen, int(b2[1])))
                            else:
                                if oy > 0 and cx2 > cx1 and dist < min_x: min_x = r_dist = dist; r_pts = ((int(cx1), int(cy1)), (int(cx2), int(cy2)))
                                if ox > 0 and cy2 > cy1 and dist < min_y: min_y = b_dist = dist; b_pts = ((int(cx1), int(cy1)), (int(cx2), int(cy2)))
                        if is_knn:
                            for d, pt1, pt2 in sorted(knn_cands, key=lambda x: x[0])[:c["n_neighbors"]]:
                                cv2.line(plotted_img, pt1, pt2, c1, 2); cv2.putText(plotted_img, f"{d:.1f}", (int((pt1[0]+pt2[0])/2), int((pt1[1]+pt2[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c1, 2); measured_distances.append(round(d, 1))
                        else:
                            if r_pts: cv2.line(plotted_img, r_pts[0], r_pts[1], c1, 2); cv2.putText(plotted_img, f"{r_dist:.1f}", (int((r_pts[0][0]+r_pts[1][0])/2), int((r_pts[0][1]+r_pts[1][1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c1, 2); measured_distances.append(round(r_dist, 1))
                            if b_pts: cv2.line(plotted_img, b_pts[0], b_pts[1], c2, 2); cv2.putText(plotted_img, f"{b_dist:.1f}", (int((b_pts[0][0]+b_pts[1][0])/2), int((b_pts[0][1]+b_pts[1][1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c2, 2); measured_distances.append(round(b_dist, 1))
                
                # OpenCV 한글 경로 저장 문제 해결을 위해 imencode 사용
                is_success, im_buf_arr = cv2.imencode(".jpg", plotted_img)
                if is_success:
                    im_buf_arr.tofile(str(result_img_path))
                    
                image_pairs.append({"파일명": img_name, "탐지 수": f"{num_objects}개", "최저 신뢰도": round(worst_conf, 2), dist_col_name: ", ".join([str(d) for d in sorted(list(set(measured_distances)))]) if measured_distances else "측정된 선 없음", "_img_path": str(result_img_path)})
            
            df_export = pd.DataFrame([{k:v for k,v in item.items() if k!="_img_path"} for item in image_pairs])
            df_export.to_csv(final_save_dir / f"{dist_run_name}_results.csv", index=False, encoding="utf-8-sig")
            thresholds = [float(x.strip()) for x in c["edge_thresholds"].split(",") if x.strip()] if is_edge else [60.0]
            def get_cat(d, thr):
                if d < thr[0]: return f"유형 1 ( < {thr[0]} )"
                for i in range(len(thr)-1):
                    if thr[i] <= d < thr[i+1]: return f"유형 {i+2} ( {thr[i]}~{thr[i+1]} )"
                return f"유형 {len(thr)+1} ( >= {thr[-1]} )"
            all_data = []
            for _, row in df_export.iterrows():
                dstr = str(row[dist_col_name])
                if dstr != "nan" and "없음" not in dstr:
                    for d in [float(x.strip()) for x in dstr.split(",")]: all_data.append({"파일명": row["파일명"], "구분": get_cat(d, thresholds) if is_edge else "유클리드", "거리(px)": d})
            df_parsed = pd.DataFrame(all_data); outliers_list = []
            if not df_parsed.empty:
                stats_list = []  
                for cat in df_parsed["구분"].unique():
                    subset = df_parsed[df_parsed["구분"] == cat]
                    Q1, Q3 = subset["거리(px)"].quantile(0.25), subset["거리(px)"].quantile(0.75); IQR = Q3 - Q1
                    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                    outs = subset[(subset["거리(px)"] < lo) | (subset["거리(px)"] > hi)].copy()
                    if not outs.empty: outs["하한"], outs["상한"] = round(lo, 2), round(hi, 2); outliers_list.append(outs)
                    stats_list.append({"구분(유형)": cat, "데이터 수(Count)": len(subset), "평균(Mean)": round(subset["거리(px)"].mean(), 2), "표준편차(Std)": round(subset["거리(px)"].std(), 2) if len(subset) > 1 else 0.0, "최소값(Min)": round(subset["거리(px)"].min(), 2), "1사분위(Q1)": round(Q1, 2), "중앙값(Median)": round(subset["거리(px)"].median(), 2), "3사분위(Q3)": round(Q3, 2), "최대값(Max)": round(subset["거리(px)"].max(), 2), "IQR": round(IQR, 2), "정상 하한값(Lower Bounds)": round(lo, 2), "정상 상한값(Upper Bounds)": round(hi, 2), "이상치 개수": len(outs)})
                pd.DataFrame(stats_list).to_csv(final_save_dir / f"{dist_run_name}_statistics.csv", index=False, encoding="utf-8-sig")
                if outliers_list: pd.concat(outliers_list, ignore_index=True).to_csv(final_save_dir / f"{dist_run_name}_outliers.csv", index=False, encoding="utf-8-sig")
            df_outliers = pd.concat(outliers_list, ignore_index=True) if outliers_list else pd.DataFrame()
            del model; clear_vram(); 
            logger.info(f"[Measure Thread] 측정 완료. 결과 건수: {len(df_export)}, 이상치: {len(df_outliers)}")
            self.finished_ok.emit(df_export, df_parsed, df_outliers, image_pairs)
        except Exception as e: 
            logger.error("[Measure Thread] 에러 발생", exc_info=True)
            self.error.emit(traceback.format_exc())

class LabelDialog(QDialog):
    def __init__(self, class_names, parent=None):
        super().__init__(parent); self.setWindowTitle("클래스 선택"); self.class_id = -1
        layout = QVBoxLayout(self); self.combo = QComboBox(); self.combo.addItems(class_names)
        layout.addWidget(QLabel("객체의 클래스를 선택하세요:")); layout.addWidget(self.combo)
        btn_layout = QHBoxLayout(); btn_ok = QPushButton("확인"); btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("취소"); btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok); btn_layout.addWidget(btn_cancel); layout.addLayout(btn_layout)
    def get_class_index(self): return self.combo.currentIndex()

class LabelingView(QGraphicsView):
    box_added = pyqtSignal(list); request_prev = pyqtSignal(); request_next = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent); self.scene = QGraphicsScene(self); self.setScene(self.scene); self.setRenderHint(QPainter.Antialiasing); self.setMouseTracking(True); self.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.crosshair_color = QColor(255, 255, 255, 180); self.image_item = None; self.current_rect_item = None; self.pending_rect_item = None; self.start_pos = None
        self.class_names = []; self.boxes = []; self.selected_indices = []; self.undo_stack = []; self.redo_stack = []; self.pre_move_state = []
        self.is_panning = False; self.last_mouse_pos = None; self.is_moving_box = False; self.moving_box_data = None; self.move_start_pos = None; self.original_rect = None
        self.is_resizing_box = False; self.resize_handle = None; self.resize_target_data = None; self.setFocusPolicy(Qt.StrongFocus)

    def load_image(self, img_path, class_names):
        self.scene.clear(); self.boxes.clear(); self.undo_stack.clear(); self.redo_stack.clear()
        self.pending_rect_item = None; self.current_rect_item = None; self.class_names = class_names; self.selected_indices = []
        self.is_resizing_box = False; self.resize_handle = None; self.resize_target_data = None; self.current_image_path = img_path
        pixmap = QPixmap(img_path)
        if pixmap.isNull(): return False
        self.image_item = QGraphicsPixmapItem(pixmap); self.scene.addItem(self.image_item); self.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.h_line = QGraphicsLineItem(); self.v_line = QGraphicsLineItem()
        pen = QPen(self.crosshair_color, 1, Qt.DashLine); self.h_line.setPen(pen); self.v_line.setPen(pen)
        self.h_line.setZValue(9999); self.v_line.setZValue(9999); self.scene.addItem(self.h_line); self.scene.addItem(self.v_line); self.h_line.hide(); self.v_line.hide()
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio); return True

    def get_resize_handle(self, pos, rect, margin=8):
        if not rect.adjusted(-margin, -margin, margin, margin).contains(pos): return None
        left, right, top, bottom = abs(pos.x() - rect.left()) < margin, abs(pos.x() - rect.right()) < margin, abs(pos.y() - rect.top()) < margin, abs(pos.y() - rect.bottom()) < margin
        if left and top: return 'TL'
        if right and top: return 'TR'
        if left and bottom: return 'BL'
        if right and bottom: return 'BR'
        if left and rect.top() <= pos.y() <= rect.bottom(): return 'L'
        if right and rect.top() <= pos.y() <= rect.bottom(): return 'R'
        if top and rect.left() <= pos.x() <= rect.right(): return 'T'
        if bottom and rect.left() <= pos.x() <= rect.right(): return 'B'
        return None

    def save_state(self): self.undo_stack.append(self.get_yolo_format()); self.redo_stack.clear()
    def clear_boxes(self):
        for rect_item, text_item, _ in self.boxes:
            if rect_item.scene() == self.scene: self.scene.removeItem(rect_item)
            if text_item.scene() == self.scene: self.scene.removeItem(text_item)
        self.boxes.clear(); self.selected_indices = []
    def apply_state(self, state):
        self.clear_boxes()
        if state: self.load_existing_labels(state)
        self.box_added.emit(self.get_yolo_format())
    def wheelEvent(self, event):
        if not self.image_item: return
        zoom_in_factor = 1.15; zoom_out_factor = 1 / zoom_in_factor
        old_pos = self.mapToScene(event.pos())
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom_factor, zoom_factor); new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos; self.translate(delta.x(), delta.y())

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton: 
            self.is_panning = True; self.last_mouse_pos = event.pos(); self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.MiddleButton and self.image_item: 
            pos = self.mapToScene(event.pos())
            for b in reversed(self.boxes):
                if b[0].rect().contains(pos): 
                    self.pre_move_state = self.get_yolo_format(); self.is_moving_box = True; self.moving_box_data = b; self.move_start_pos = pos; self.original_rect = b[0].rect(); self.setCursor(Qt.ClosedHandCursor); break
        elif event.button() == Qt.LeftButton and self.image_item: 
            if self.resize_handle and self.resize_target_data:
                self.pre_move_state = self.get_yolo_format(); self.is_resizing_box = True; self.original_rect = self.resize_target_data[0].rect()
            else:
                if self.pending_rect_item: self.scene.removeItem(self.pending_rect_item); self.pending_rect_item = None
                self.start_pos = self.mapToScene(event.pos()); self.current_rect_item = QGraphicsRectItem(); self.current_rect_item.setPen(QPen(Qt.red, 2, Qt.SolidLine)); self.scene.addItem(self.current_rect_item)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        cur_pos = self.mapToScene(event.pos())
        if self.image_item and hasattr(self, 'h_line'):
            if self.sceneRect().contains(cur_pos):
                self.h_line.setLine(0, cur_pos.y(), self.sceneRect().width(), cur_pos.y()); self.v_line.setLine(cur_pos.x(), 0, cur_pos.x(), self.sceneRect().height())
                self.h_line.show(); self.v_line.show()
            else: self.h_line.hide(); self.v_line.hide()
        if not (event.buttons() & Qt.LeftButton) and not (event.buttons() & Qt.MiddleButton) and not (event.buttons() & Qt.RightButton):
            cursor = Qt.CrossCursor; self.resize_target_data = None; self.resize_handle = None
            for b in reversed(self.boxes):
                rect = b[0].rect()
                handle = self.get_resize_handle(cur_pos, rect)
                if handle:
                    self.resize_target_data = b; self.resize_handle = handle
                    if handle in ('TL', 'BR'): cursor = Qt.SizeFDiagCursor
                    elif handle in ('TR', 'BL'): cursor = Qt.SizeBDiagCursor
                    elif handle in ('L', 'R'): cursor = Qt.SizeHorCursor
                    elif handle in ('T', 'B'): cursor = Qt.SizeVerCursor
                    break
            self.viewport().setCursor(cursor)
        if self.is_panning:
            delta = event.pos() - self.last_mouse_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x()); self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y()); self.last_mouse_pos = event.pos()
        elif self.is_moving_box and self.moving_box_data: 
            delta = cur_pos - self.move_start_pos
            new_rect = self.original_rect.translated(delta.x(), delta.y())
            new_x = max(0, min(new_rect.x(), self.sceneRect().width() - new_rect.width())); new_y = max(0, min(new_rect.y(), self.sceneRect().height() - new_rect.height()))
            new_rect.moveTo(new_x, new_y)
            self.moving_box_data[0].setRect(new_rect); self.moving_box_data[1].setPos(new_rect.topLeft())
        elif self.is_resizing_box and self.resize_target_data: 
            rect = QRectF(self.original_rect)
            if 'L' in self.resize_handle: rect.setLeft(min(cur_pos.x(), rect.right() - 5))
            if 'R' in self.resize_handle: rect.setRight(max(cur_pos.x(), rect.left() + 5))
            if 'T' in self.resize_handle: rect.setTop(min(cur_pos.y(), rect.bottom() - 5))
            if 'B' in self.resize_handle: rect.setBottom(max(cur_pos.y(), rect.top() + 5))
            self.resize_target_data[0].setRect(rect); self.resize_target_data[1].setPos(rect.topLeft())
        elif self.start_pos and self.current_rect_item:
            rect = QRectF(self.start_pos, cur_pos).normalized(); rect = rect.intersected(self.sceneRect()); self.current_rect_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton: self.is_panning = False; self.viewport().setCursor(Qt.CrossCursor)
        elif event.button() == Qt.MiddleButton and self.is_moving_box: 
            self.is_moving_box = False; self.moving_box_data = None; self.viewport().setCursor(Qt.CrossCursor)
            current_state = self.get_yolo_format()
            if self.pre_move_state != current_state: self.undo_stack.append(self.pre_move_state); self.redo_stack.clear()
            self.box_added.emit(current_state)
        elif event.button() == Qt.LeftButton:
            if self.is_resizing_box:
                self.is_resizing_box = False; current_state = self.get_yolo_format()
                if self.pre_move_state != current_state: self.undo_stack.append(self.pre_move_state); self.redo_stack.clear()
                self.box_added.emit(current_state); self.resize_target_data = None; self.resize_handle = None
            elif self.current_rect_item:
                rect = self.current_rect_item.rect()
                if rect.width() > 5 and rect.height() > 5: self.pending_rect_item = self.current_rect_item; self.pending_rect_item.setPen(QPen(Qt.yellow, 2, Qt.DashLine))
                else: self.scene.removeItem(self.current_rect_item)
                self.current_rect_item = None; self.start_pos = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if hasattr(self, 'h_line'): self.h_line.hide(); self.v_line.hide()
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A: self.request_prev.emit()
        elif event.key() == Qt.Key_D: self.request_next.emit()
        elif event.key() == Qt.Key_Q:
            if self.pending_rect_item: self.scene.removeItem(self.pending_rect_item); self.pending_rect_item = None
            elif self.undo_stack: self.redo_stack.append(self.get_yolo_format()); self.apply_state(self.undo_stack.pop())
        elif event.key() == Qt.Key_E:
            if self.redo_stack: self.undo_stack.append(self.get_yolo_format()); self.apply_state(self.redo_stack.pop())
        elif event.key() == Qt.Key_S:
            if self.pending_rect_item and len(self.class_names) > 0: self.finalize_pending_box(0)
        elif event.key() == Qt.Key_W:
            if self.pending_rect_item and len(self.class_names) > 1: self.finalize_pending_box(1)
        elif event.key() == Qt.Key_Escape:
            if self.pending_rect_item: self.scene.removeItem(self.pending_rect_item); self.pending_rect_item = None
        elif event.key() in (Qt.Key_Space, Qt.Key_Return):
            if self.pending_rect_item:
                dialog = LabelDialog(self.class_names, self)
                if dialog.exec_() == QDialog.Accepted: self.finalize_pending_box(dialog.get_class_index())
                else: self.scene.removeItem(self.pending_rect_item); self.pending_rect_item = None
        else: super().keyPressEvent(event)

    def finalize_pending_box(self, class_id):
        if not self.pending_rect_item: return
        self.save_state(); class_name = self.class_names[class_id]
        color = Qt.green if class_id == 0 else Qt.red
        self.pending_rect_item.setPen(QPen(color, 2, Qt.SolidLine))
        text_item = self.scene.addText(class_name, QFont("Arial", 12, QFont.Bold)); text_item.setDefaultTextColor(color); text_item.setPos(self.pending_rect_item.rect().topLeft())
        self.boxes.append((self.pending_rect_item, text_item, class_id)); self.box_added.emit(self.get_yolo_format()); self.pending_rect_item = None

    def get_yolo_format(self):
        if not self.image_item: return []
        img_w, img_h = self.sceneRect().width(), self.sceneRect().height(); yolo_data = []
        for rect_item, text_item, class_id in self.boxes:
            rect = rect_item.rect()
            yolo_data.append(f"{class_id} {(rect.x() + rect.width() / 2) / img_w:.6f} {(rect.y() + rect.height() / 2) / img_h:.6f} {rect.width() / img_w:.6f} {rect.height() / img_h:.6f}")
        return yolo_data
    
    def set_crosshair_color(self, qcolor):
        self.crosshair_color = qcolor
        if hasattr(self, 'h_line') and hasattr(self, 'v_line'):
            pen = QPen(self.crosshair_color, 1, Qt.DashLine); self.h_line.setPen(pen); self.v_line.setPen(pen)
    
    def load_existing_labels(self, yolo_lines):
        if not self.image_item: return
        img_w, img_h = self.sceneRect().width(), self.sceneRect().height()
        for line in yolo_lines:
            parts = line.split()
            if len(parts) == 5:
                try:
                    class_id = int(parts[0]); x_center, y_center, norm_w, norm_h = map(float, parts[1:5])
                    w, h = norm_w * img_w, norm_h * img_h; x, y = (x_center * img_w) - (w / 2), (y_center * img_h) - (h / 2)
                    rect = QRectF(x, y, w, h); rect_item = QGraphicsRectItem(rect)
                    color = Qt.green if class_id == 0 else Qt.red; rect_item.setPen(QPen(color, 2, Qt.SolidLine))
                    class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"Class {class_id}"
                    text_item = self.scene.addText(class_name, QFont("Arial", 12, QFont.Bold)); text_item.setDefaultTextColor(color); text_item.setPos(rect.topLeft())
                    self.scene.addItem(rect_item); self.boxes.append((rect_item, text_item, class_id))
                except Exception: pass
        self.box_added.emit(self.get_yolo_format())

    def delete_boxes(self, indices):
        if not indices: return
        self.save_state()
        for index in sorted(indices, reverse=True):
            if 0 <= index < len(self.boxes):
                rect_item, text_item, _ = self.boxes.pop(index); self.scene.removeItem(rect_item); self.scene.removeItem(text_item)
        self.box_added.emit(self.get_yolo_format())

    def update_boxes_class(self, indices, new_class_id):
        if not indices: return
        self.save_state()
        for index in indices:
            if 0 <= index < len(self.boxes):
                rect_item, text_item, _ = self.boxes[index]
                class_name = self.class_names[new_class_id]; color = Qt.green if new_class_id == 0 else Qt.red
                rect_item.setPen(QPen(color, 2, Qt.SolidLine)); text_item.setPlainText(class_name); text_item.setDefaultTextColor(color); self.boxes[index] = (rect_item, text_item, new_class_id)
        self.box_added.emit(self.get_yolo_format())

    def select_boxes(self, indices):
        for i, (rect_item, text_item, class_id) in enumerate(self.boxes):
            color = Qt.green if class_id == 0 else Qt.red; rect_item.setPen(QPen(color, 2, Qt.SolidLine)); text_item.setDefaultTextColor(color)
        self.selected_indices = indices
        for idx in self.selected_indices:
            if 0 <= idx < len(self.boxes):
                rect_item, text_item, class_id = self.boxes[idx]
                rect_item.setPen(QPen(Qt.cyan, 4, Qt.SolidLine)); text_item.setDefaultTextColor(Qt.cyan)

    def auto_label_similar(self, threshold, nms_threshold):
        if not hasattr(self, 'selected_indices') or len(self.selected_indices) != 1: return False, "기준 박스를 1개만 선택해주세요."
        idx = self.selected_indices[0]
        if idx < 0 or idx >= len(self.boxes): return False, "유효하지 않은 선택입니다."
        if not self.image_item or not hasattr(self, 'current_image_path'): return False, "이미지가 없습니다."
        
        # OpenCV 한글 경로 읽기 문제 해결
        img_array = np.fromfile(str(self.current_image_path), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None: return False, "이미지를 읽을 수문을 읽을 수 없습니다."
        rect_item, _, class_id = self.boxes[idx]
        r = rect_item.rect(); x, y, w, h = int(r.x()), int(r.y()), int(r.width()), int(r.height())
        if w < 5 or h < 5 or x < 0 or y < 0 or x+w > img.shape[1] or y+h > img.shape[0]: return False, "크기/위치가 유효하지 않습니다."
        template = img[y:y+h, x:x+w].copy() 
        try:
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            cand_boxes, scores = [], []
            for pt in zip(*loc[::-1]): cand_boxes.append([pt[0], pt[1], w, h]); scores.append(float(res[pt[1], pt[0]]))
            if not cand_boxes: return True, 0 
            indices = cv2.dnn.NMSBoxes(cand_boxes, scores, threshold, nms_threshold)
            flat_indices = np.array(indices).flatten() if len(indices) > 0 else []
            existing_boxes = [(int(b[0].rect().x()), int(b[0].rect().y()), int(b[0].rect().width()), int(b[0].rect().height())) for b in self.boxes]
            self.save_state(); added_count = 0
            for idx in flat_indices:
                nx, ny, nw, nh = cand_boxes[int(idx)]; cx, cy = nx + nw/2, ny + nh/2; is_dup = False
                for ex, ey, ew, eh in existing_boxes:
                    if abs(cx - (ex + ew/2)) < nw/2 and abs(cy - (ey + eh/2)) < nh/2: is_dup = True; break
                if is_dup: continue
                new_rect = QRectF(nx, ny, nw, nh); new_rect_item = QGraphicsRectItem(new_rect)
                color = Qt.green if class_id == 0 else Qt.red; new_rect_item.setPen(QPen(color, 2, Qt.SolidLine))
                text_item = self.scene.addText(self.class_names[class_id], QFont("Arial", 12, QFont.Bold)); text_item.setDefaultTextColor(color); text_item.setPos(new_rect.topLeft())
                self.scene.addItem(new_rect_item); self.boxes.append((new_rect_item, text_item, class_id)); added_count += 1
            if added_count > 0: self.box_added.emit(self.get_yolo_format()) 
            return True, added_count
        finally:
            del template
            if 'res' in locals(): del res

    def save_all_templates(self):
        if not self.image_item or not hasattr(self, 'current_image_path'): return False
        
        # OpenCV 한글 경로 읽기 문제 해결
        import cv2; import numpy as np
        img_array = np.fromfile(str(self.current_image_path), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None: return False
        self.saved_templates = []
        for rect_item, _, class_id in self.boxes:
            r = rect_item.rect(); x, y, w, h = int(r.x()), int(r.y()), int(r.width()), int(r.height())
            if w < 5 or h < 5 or x < 0 or y < 0 or x+w > img.shape[1] or y+h > img.shape[0]: continue
            self.saved_templates.append({'template': img[y:y+h, x:x+w].copy(), 'class_id': class_id, 'w': w, 'h': h})
        return len(self.saved_templates) > 0

    def auto_label_from_saved_templates(self, threshold, nms_threshold):
        if not hasattr(self, 'saved_templates') or not self.saved_templates: return False, "저장된 기준 객체가 없습니다."
        if not self.image_item or not hasattr(self, 'current_image_path'): return False, "이미지가 로드되지 않았습니다."
        
        # OpenCV 한글 경로 읽기 문제 해결
        import cv2, numpy as np; 
        img_array = np.fromfile(str(self.current_image_path), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None: return False, "이미지를 읽을 수 없습니다."
        self.save_state(); total_added = 0
        for tmpl_data in self.saved_templates:
            template, class_id, w, h = tmpl_data['template'], class_id, tmpl_data['w'], tmpl_data['h']
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            cand_boxes, scores = [], []
            for pt in zip(*loc[::-1]): cand_boxes.append([pt[0], pt[1], w, h]); scores.append(float(res[pt[1], pt[0]]))
            if not cand_boxes: continue
            indices = cv2.dnn.NMSBoxes(cand_boxes, scores, threshold, nms_threshold)
            flat_indices = np.array(indices).flatten() if len(indices) > 0 else []
            existing_boxes = [(int(b[0].rect().x()), int(b[0].rect().y()), int(b[0].rect().width()), int(b[0].rect().height())) for b in self.boxes]
            for idx in flat_indices:
                nx, ny, nw, nh = cand_boxes[int(idx)]; cx, cy = nx + nw/2, ny + nh/2; is_dup = False
                for ex, ey, ew, eh in existing_boxes:
                    if abs(cx - (ex + ew/2)) < nw/2 and abs(cy - (ey + eh/2)) < nh/2: is_dup = True; break
                if is_dup: continue
                new_rect = QRectF(nx, ny, nw, nh); new_rect_item = QGraphicsRectItem(new_rect)
                color = Qt.green if class_id == 0 else Qt.red; new_rect_item.setPen(QPen(color, 2, Qt.SolidLine))
                text_item = self.scene.addText(self.class_names[class_id], QFont("Arial", 12, QFont.Bold)); text_item.setDefaultTextColor(color); text_item.setPos(new_rect.topLeft())
                self.scene.addItem(new_rect_item); self.boxes.append((new_rect_item, text_item, class_id)); existing_boxes.append((int(nx), int(ny), int(nw), int(nh))); total_added += 1
        if total_added > 0: self.box_added.emit(self.get_yolo_format()) 
        return True, total_added

class PathInputWidget(QWidget):
    def __init__(self, label, is_folder=True, default_path=""):
        super().__init__(); self.is_folder = is_folder; layout = QHBoxLayout(); layout.setContentsMargins(0, 0, 0, 0)
        self.line_edit = QLineEdit(default_path); self.btn = QPushButton("📂"); self.btn.clicked.connect(self.browse)
        layout.addWidget(QLabel(label)); layout.addWidget(self.line_edit); layout.addWidget(self.btn); self.setLayout(layout)
    def browse(self):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택", self.line_edit.text()) if self.is_folder else QFileDialog.getOpenFileName(self, "파일 선택", self.line_edit.text(), "PyTorch Model (*.pt);;All Files (*)")[0]
        if path: self.line_edit.setText(path)
    def get_path(self): return self.line_edit.text()

class ImagePreviewDialog(QDialog):
    def __init__(self, image_paths, start_index, parent=None):
        super().__init__(parent); self.image_paths = image_paths; self.current_index = start_index; self.original_pixmap = None; self.scale_factor = 1.0; self.is_panning = False; self.last_mouse_pos = None
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(10, 10, 10, 10)
        self.filename_edit = QLineEdit(); self.filename_edit.setReadOnly(True); self.filename_edit.setStyleSheet("background: transparent; border: none; font-size: 14px; font-weight: bold; color: #333;"); self.filename_edit.setFocusPolicy(Qt.ClickFocus); self.layout.addWidget(self.filename_edit)
        self.layout.addWidget(QLabel("⬅️ ➡️ 방향키: 이전/다음 | 🖱️ 마우스 휠: 확대/축소 | ✋ 클릭 후 드래그: 화면 이동"))
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFocusPolicy(Qt.NoFocus)
        self.image_label = QLabel(); self.image_label.setAlignment(Qt.AlignCenter); self.image_label.setCursor(Qt.OpenHandCursor); self.scroll.setWidget(self.image_label); self.layout.addWidget(self.scroll)
        self.image_label.installEventFilter(self); self.setFocusPolicy(Qt.StrongFocus); self.load_image(); self.setFocus()

    def load_image(self):
        if not (0 <= self.current_index < len(self.image_paths)): return
        path = self.image_paths[self.current_index]; file_name = Path(path).name; self.setWindowTitle(f"이미지 상세 보기 ({self.current_index + 1}/{len(self.image_paths)})"); self.filename_edit.setText(file_name); self.original_pixmap = QPixmap(path)
        if not self.original_pixmap.isNull():
            screen_geo = QApplication.primaryScreen().availableGeometry(); max_w, max_h = screen_geo.width() - 100, screen_geo.height() - 150; img_w, img_h = self.original_pixmap.width(), self.original_pixmap.height()
            self.scale_factor = min(max_w / img_w, max_h / img_h) if img_w > max_w or img_h > max_h else 1.0 
            self.update_image_display(); self.resize(int(img_w * self.scale_factor) + 40, int(img_h * self.scale_factor) + 100)

    def update_image_display(self):
        if self.original_pixmap and not self.original_pixmap.isNull(): self.image_label.setPixmap(self.original_pixmap.scaled(int(self.original_pixmap.width() * self.scale_factor), int(self.original_pixmap.height() * self.scale_factor), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def wheelEvent(self, event):
        if self.original_pixmap is None or self.original_pixmap.isNull(): return
        h_bar, v_bar = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
        h_ratio = h_bar.value() / h_bar.maximum() if h_bar.maximum() > 0 else 0; v_ratio = v_bar.value() / v_bar.maximum() if v_bar.maximum() > 0 else 0
        if event.angleDelta().y() > 0: self.scale_factor *= 1.15
        elif event.angleDelta().y() < 0: self.scale_factor /= 1.15
        self.scale_factor = max(0.05, min(self.scale_factor, 10.0)); self.update_image_display()
        h_bar.setValue(int(h_bar.maximum() * h_ratio)); v_bar.setValue(int(v_bar.maximum() * v_ratio))

    def eventFilter(self, source, event):
        if source == self.image_label:
            if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton: self.is_panning = True; self.last_mouse_pos = event.globalPos(); self.image_label.setCursor(Qt.ClosedHandCursor); return True
            elif event.type() == event.MouseMove and self.is_panning: delta = event.globalPos() - self.last_mouse_pos; self.last_mouse_pos = event.globalPos(); self.scroll.horizontalScrollBar().setValue(self.scroll.horizontalScrollBar().value() - delta.x()); self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().value() - delta.y()); return True
            elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton: self.is_panning = False; self.image_label.setCursor(Qt.OpenHandCursor); return True
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            if self.current_index > 0: self.current_index -= 1; self.load_image()
        elif event.key() == Qt.Key_Right:
            if self.current_index < len(self.image_paths) - 1: self.current_index += 1; self.load_image()
        else: super().keyPressEvent(event)

class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)
    def __init__(self, path): super().__init__(); self.path = path; self.setCursor(Qt.PointingHandCursor); self.setAlignment(Qt.AlignCenter); self.original_pixmap = QPixmap(path); self.setMinimumSize(150, 150); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    def resizeEvent(self, event):
        if hasattr(self, 'original_pixmap') and not self.original_pixmap.isNull(): self.setPixmap(self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.clicked.emit(self.path)
    def hasHeightForWidth(self): return True
    def heightForWidth(self, width): return int(width * (self.original_pixmap.height() / self.original_pixmap.width())) if not self.original_pixmap.isNull() and self.original_pixmap.width() > 0 else width
    def sizeHint(self): return QSize(400, self.heightForWidth(400))

class ImageGridWidget(QWidget):
    def __init__(self, max_display=100):
        super().__init__(); self.max_display = max_display; self.current_page = 0; self.all_image_paths = []; self.current_images = []; layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.page_controls = QWidget(); pc_layout = QHBoxLayout(self.page_controls); pc_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_prev = QPushButton("◀ 이전"); self.btn_prev.clicked.connect(self.prev_page); self.lbl_page = QLabel("1 / 1 페이지 (총 0장)"); self.lbl_page.setAlignment(Qt.AlignCenter); self.btn_next = QPushButton("다음 ▶"); self.btn_next.clicked.connect(self.next_page)
        pc_layout.addWidget(self.btn_prev); pc_layout.addWidget(self.lbl_page); pc_layout.addWidget(self.btn_next)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.container = QWidget(); self.grid = QGridLayout(self.container); self.grid.setSpacing(15); self.scroll.setWidget(self.container)
        layout.addWidget(self.page_controls); layout.addWidget(self.scroll)

    def update_images(self, image_paths): self.all_image_paths = image_paths; self.current_page = 0; self.show_page()

    def show_page(self):
        total_items = len(self.all_image_paths); total_pages = max(1, math.ceil(total_items / self.max_display)); self.lbl_page.setText(f"{self.current_page + 1} / {total_pages} 페이지 (총 {total_items}장)")
        self.btn_prev.setEnabled(self.current_page > 0); self.btn_next.setEnabled(self.current_page < total_pages - 1)
        start_idx = self.current_page * self.max_display; end_idx = start_idx + self.max_display; self.current_images = self.all_image_paths[start_idx:end_idx]
        for i in reversed(range(self.grid.count())): 
            if self.grid.itemAt(i).widget(): self.grid.itemAt(i).widget().setParent(None)
        for i in range(self.grid.rowCount()): self.grid.setRowStretch(i, 0)
        for i in range(self.grid.columnCount()): self.grid.setColumnStretch(i, 0)
        self.grid.setColumnStretch(0, 1); self.grid.setColumnStretch(1, 1)
        for i, path in enumerate(self.current_images): lbl = ClickableLabel(path); lbl.clicked.connect(self.show_full_image); self.grid.addWidget(lbl, i // 2, i % 2)
        self.grid.setRowStretch((len(self.current_images) + 1) // 2, 1); self.scroll.verticalScrollBar().setValue(0)

    def prev_page(self):
        if self.current_page > 0: self.current_page -= 1; self.show_page()

    def next_page(self):
        if self.current_page < math.ceil(len(self.all_image_paths) / self.max_display) - 1: self.current_page += 1; self.show_page()

    def show_full_image(self, path):
        if path in self.current_images: ImagePreviewDialog(self.current_images, self.current_images.index(path), self).exec_()

class LogTabWidget(QWidget):
    def __init__(self, tab_type, db_manager, parent=None):
        super().__init__(parent)
        self.tab_type = tab_type # 'train' or 'eval'
        self.db_manager = db_manager
        self.current_page = 1
        self.page_size = 50
        self.sort_col = "id"
        self.sort_order = "DESC"
        self.init_ui()
        self.load_data()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Top Filter Area ---
        filter_layout = QHBoxLayout()
        
        self.dt_from = QDateEdit(QDate.currentDate().addDays(-30))
        self.dt_from.setCalendarPopup(True)
        self.dt_to = QDateEdit(QDate.currentDate().addDays(1))
        self.dt_to.setCalendarPopup(True)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("모델명 또는 유형 검색")
        self.search_input.returnPressed.connect(self.on_search)
        
        self.chk_ng_only = QCheckBox("NG(오답) 항목만 보기")
        self.chk_ng_only.setVisible(self.tab_type == 'eval')
        
        btn_search = QPushButton("🔍 조회/새로고침")
        btn_search.clicked.connect(self.on_search)
        
        btn_export = QPushButton("📥 CSV 내보내기")
        btn_export.clicked.connect(self.export_csv)
        
        btn_delete = QPushButton("🗑️ 선택 삭제")
        btn_delete.clicked.connect(self.delete_selected)

        filter_layout.addWidget(QLabel("기간:"))
        filter_layout.addWidget(self.dt_from)
        filter_layout.addWidget(QLabel("~"))
        filter_layout.addWidget(self.dt_to)
        filter_layout.addSpacing(10)
        filter_layout.addWidget(self.search_input)
        if self.tab_type == 'eval':
            filter_layout.addWidget(self.chk_ng_only)
        filter_layout.addWidget(btn_search)
        filter_layout.addWidget(btn_export)
        filter_layout.addWidget(btn_delete)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)

        # --- Main Splitter ---
        splitter = QSplitter(Qt.Vertical) 
        
        # [Top]: Table Area
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().sectionClicked.connect(self.on_header_clicked)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        self.table.itemDoubleClicked.connect(self.on_row_double_clicked)
        splitter.addWidget(self.table)
        
        # [Bottom]: Details Panel
        self.details_panel = QWidget()
        details_layout = QVBoxLayout(self.details_panel)
        details_layout.setContentsMargins(0, 10, 0, 0)
        
        # 상세 정보 헤더
        detail_header_layout = QHBoxLayout()
        self.lbl_detail_title = QLabel("<b>[상세 정보]</b> 목록에서 항목을 선택하세요.")
        self.btn_open_folder = QPushButton("📂 저장 폴더 열기")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self.open_current_folder)
        self.btn_open_folder.setFixedWidth(150)
        
        detail_header_layout.addWidget(self.lbl_detail_title)
        detail_header_layout.addStretch()
        detail_header_layout.addWidget(self.btn_open_folder)
        details_layout.addLayout(detail_header_layout)
        
        # 상세 정보 탭 생성
        self.detail_tabs = QTabWidget()
        
        # 1. Config 탭
        self.txt_config = QTextEdit()
        self.txt_config.setReadOnly(True)
        self.txt_config.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace; font-size: 12px;")
        self.detail_tabs.addTab(self.txt_config, "⚙️ 설정 (Config) 및 변경점")
        
        # 2. 오답 이미지 탭
        if self.tab_type == 'eval':
            self.list_wrong_imgs = QListWidget()
            self.list_wrong_imgs.itemDoubleClicked.connect(self.on_wrong_image_double_clicked)
            self.detail_tabs.addTab(self.list_wrong_imgs, "🖼️ 오답 이미지 목록")
            
        details_layout.addWidget(self.detail_tabs)
        splitter.addWidget(self.details_panel)
        splitter.setSizes([600, 400]) 
        layout.addWidget(splitter, stretch=1)
        
        # --- Bottom Paging Area ---
        paging_layout = QHBoxLayout()
        self.btn_first = QPushButton("|<")
        self.btn_prev = QPushButton("<")
        self.lbl_page = QLabel("Page 1 / 1")
        self.lbl_page.setAlignment(Qt.AlignCenter)
        self.btn_next = QPushButton(">")
        self.btn_last = QPushButton(">|")
        
        self.btn_first.clicked.connect(lambda: self.change_page(1))
        self.btn_prev.clicked.connect(lambda: self.change_page(self.current_page - 1))
        self.btn_next.clicked.connect(lambda: self.change_page(self.current_page + 1))
        
        self.cmb_page_size = QComboBox()
        self.cmb_page_size.addItems(["50개씩 보기", "100개씩 보기", "500개씩 보기"])
        self.cmb_page_size.currentIndexChanged.connect(self.on_page_size_changed)
        
        paging_layout.addStretch()
        paging_layout.addWidget(self.btn_first)
        paging_layout.addWidget(self.btn_prev)
        paging_layout.addWidget(self.lbl_page)
        paging_layout.addWidget(self.btn_next)
        paging_layout.addWidget(self.btn_last) 
        paging_layout.addWidget(self.cmb_page_size)
        paging_layout.addStretch()
        
        layout.addLayout(paging_layout)
        
        if self.tab_type == 'train':
            self.headers = ["선택", "ID", "일시", "유형", "모델명", "Epochs", "Batch", "최고mAP"]
            self.db_cols = ["id", "id", "timestamp", "task_type", "model_name", "epochs", "batch_size", "best_map"]
        else:
            self.headers = ["선택", "ID", "일시", "평가단계", "모델명", "전체(장)", "오답(장)", "정확도(%)"]
            self.db_cols = ["id", "id", "timestamp", "task_type", "model_name", "total_imgs", "wrong_count", "accuracy"]
            
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        
        self.current_rows = []

    def compare_configs(self, old_c, new_c, prefix=""):
        diffs = []
        if isinstance(old_c, dict) and isinstance(new_c, dict):
            all_keys = set(old_c.keys()).union(set(new_c.keys()))
            for k in all_keys:
                if k in ['saved_at', 'timestamp', 'name']: continue
                full_key = f"{prefix}.{k}" if prefix else k
                if k not in old_c:
                    diffs.append((full_key, "없음(None)", new_c[k]))
                elif k not in new_c:
                    diffs.append((full_key, old_c[k], "삭제됨(Deleted)"))
                else:
                    diffs.extend(self.compare_configs(old_c[k], new_c[k], full_key))
        else:
            if old_c != new_c:
                diffs.append((prefix, old_c, new_c))
        return diffs

    def on_search(self):
        self.current_page = 1
        self.load_data()

    def on_page_size_changed(self):
        val = int(self.cmb_page_size.currentText().replace("개씩 보기", ""))
        self.page_size = val
        self.current_page = 1
        self.load_data()

    def change_page(self, page):
        self.current_page = page
        self.load_data()

    def on_header_clicked(self, logical_index):
        if logical_index == 0: return 
        col_name = self.db_cols[logical_index]
        if self.sort_col == col_name:
            self.sort_order = "ASC" if self.sort_order == "DESC" else "DESC"
        else:
            self.sort_col = col_name
            self.sort_order = "DESC"
        self.load_data()

    def load_data(self):
        kw = self.search_input.text()
        d_from = self.dt_from.date().toString("yyyy-MM-dd")
        d_to = self.dt_to.date().toString("yyyy-MM-dd")
        filter_ng = self.chk_ng_only.isChecked() if self.tab_type == 'eval' else False
        
        offset = (self.current_page - 1) * self.page_size
        
        rows, total_count = self.db_manager.fetch_logs(
            table_type=self.tab_type, search_kw=kw, date_from=d_from, date_to=d_to,
            filter_ng=filter_ng, offset=offset, limit=self.page_size,
            sort_col=self.sort_col, sort_order=self.sort_order
        )
        
        self.current_rows = rows
        self.table.setRowCount(len(rows))
        
        for r_idx, row in enumerate(rows):
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Unchecked)
            self.table.setItem(r_idx, 0, chk_item)
            
            for c_idx in range(1, len(self.headers)):
                db_idx = c_idx - 1 
                val = row[db_idx]
                
                if self.tab_type == 'train' and c_idx == 7: 
                    val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                elif self.tab_type == 'eval' and c_idx == 7: 
                    val_str = f"{val:.1f}%" if isinstance(val, float) else str(val)
                else:
                    val_str = str(val)
                    
                item = QTableWidgetItem(val_str)
                item.setTextAlignment(Qt.AlignCenter)
                
                if self.tab_type == 'eval' and c_idx == 6 and val > 0: 
                    item.setForeground(QColor("red"))
                    item.setFont(QFont("Arial", 10, QFont.Bold))
                    
                self.table.setItem(r_idx, c_idx, item)
                
        total_pages = max(1, math.ceil(total_count / self.page_size))
        self.lbl_page.setText(f"Page {self.current_page} / {total_pages} (총 {total_count}건)")
        self.btn_prev.setEnabled(self.current_page > 1)
        self.btn_first.setEnabled(self.current_page > 1)
        self.btn_next.setEnabled(self.current_page < total_pages)
        self.btn_last.setEnabled(self.current_page < total_pages)
        
        try: self.btn_last.clicked.disconnect()
        except: pass
        self.btn_last.clicked.connect(lambda checked=False, p=total_pages: self.change_page(p))
        
        self.clear_details()

    def delete_selected(self):
        ids_to_delete = []
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).checkState() == Qt.Checked:
                ids_to_delete.append(int(self.table.item(r, 1).text()))
                
        if not ids_to_delete:
            QMessageBox.warning(self, "경고", "삭제할 항목을 체크해주세요.")
            return
            
        if QMessageBox.question(self, "삭제 확인", f"선택한 {len(ids_to_delete)}개의 기록을 삭제하시겠습니까?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.db_manager.delete_logs(self.tab_type, ids_to_delete)
            self.load_data()

    def export_csv(self):
        kw = self.search_input.text()
        d_from = self.dt_from.date().toString("yyyy-MM-dd")
        d_to = self.dt_to.date().toString("yyyy-MM-dd")
        filter_ng = self.chk_ng_only.isChecked() if self.tab_type == 'eval' else False
        
        rows, _ = self.db_manager.fetch_logs(
            table_type=self.tab_type, search_kw=kw, date_from=d_from, date_to=d_to,
            filter_ng=filter_ng, offset=0, limit=-1, sort_col=self.sort_col, sort_order=self.sort_order
        )
        
        if not rows:
            QMessageBox.warning(self, "경고", "내보낼 데이터가 없습니다.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "CSV 저장", f"logs_{self.tab_type}_{datetime.now().strftime('%Y%m%d')}.csv", "CSV Files (*.csv)")
        if not path: return
        
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                if self.tab_type == 'train':
                    writer.writerow(["ID", "Timestamp", "Task Type", "Model Name", "Epochs", "Batch Size", "Best mAP", "Save Dir", "Config JSON"])
                else:
                    writer.writerow(["ID", "Timestamp", "Task Type", "Model Name", "Total Imgs", "Wrong Count", "Accuracy", "Wrong Images", "Config JSON"])
                writer.writerows(rows)
            logger.info(f"CSV 파일 내보내기 성공: {path}")
            QMessageBox.information(self, "성공", f"CSV 파일이 저장되었습니다:\n{path}")
        except Exception as e:
            logger.error(f"CSV 파일 내보내기 실패: {e}", exc_info=True)
            QMessageBox.critical(self, "오류", f"CSV 저장 실패:\n{e}")

    def on_row_selected(self):
        selected = self.table.selectedItems()
        if not selected: return
        r = selected[0].row()
        row_data = self.current_rows[r]
        
        save_dir = ""
        config_str = ""
        
        if self.tab_type == 'train':
            save_dir = row_data[7]
            config_str = row_data[8]
            self.lbl_detail_title.setText(f"<b>[학습 상세]</b> ID: {row_data[0]} | 모델: {row_data[3]}")
        else:
            self.lbl_detail_title.setText(f"<b>[평가 상세]</b> ID: {row_data[0]} | 모델: {row_data[3].split('|')[0].strip()}")
            config_str = row_data[8]
            wrong_imgs_str = row_data[7]
            self.list_wrong_imgs.clear()
            
            model_path_str = row_data[3].split('|')[-1].strip() if '|' in row_data[3] else row_data[3]
            try:
                model_path = Path(model_path_str)
                if model_path.name == "best_model.pt":
                    source_txt = model_path.parent / "best_model_source.txt"
                    if source_txt.exists():
                        original_path = Path(source_txt.read_text(encoding='utf-8').strip())
                        save_dir = str(original_path.parent.parent) if original_path.parent.name == "weights" else str(original_path.parent)
                    else:
                        save_dir = str(model_path.parent)
                else:
                    save_dir = str(model_path.parent.parent) if model_path.parent.name == "weights" else str(model_path.parent)
            except:
                save_dir = ""
                
            if wrong_imgs_str:
                try:
                    imgs = json.loads(wrong_imgs_str)
                    for img in imgs:
                        self.list_wrong_imgs.addItem(img)
                except: pass

        self.btn_open_folder.setProperty("target_path", save_dir)
        self.btn_open_folder.setEnabled(bool(save_dir and Path(save_dir).exists()))
        
        try:
            current_config = json.loads(config_str)
            prev_config = {}
            
            if self.sort_col == "id" and self.sort_order == "DESC" and r + 1 < len(self.current_rows):
                prev_config_str = self.current_rows[r+1][8]
                if prev_config_str: prev_config = json.loads(prev_config_str)
            elif self.sort_col == "id" and self.sort_order == "ASC" and r - 1 >= 0:
                prev_config_str = self.current_rows[r-1][8]
                if prev_config_str: prev_config = json.loads(prev_config_str)
            
            diff_text = ""
            if prev_config:
                diffs = self.compare_configs(prev_config, current_config)
                if diffs:
                    diff_text = "<div style='background-color:#2d2d2d; padding:8px; border-radius:4px; margin-bottom:10px;'>"
                    diff_text += "<b style='color:#fca5a5; font-size:13px;'>🔍 [바로 이전 기록 대비 변경점]</b><br>"
                    for k, old_v, new_v in diffs:
                        diff_text += f"&nbsp;&nbsp;• <b>{k}</b> : <span style='text-decoration:line-through; color:#9ca3af;'>{old_v}</span> ➡️ <span style='color:#34d399; font-weight:bold;'>{new_v}</span><br>"
                    diff_text += "</div>"
                else:
                    diff_text = "<div style='color:#9ca3af; margin-bottom:10px;'>💡 바로 이전 기록과 파라미터가 동일합니다.</div>"
            
            pretty_json = json.dumps(current_config, indent=4, ensure_ascii=False)
            escaped_json = html.escape(pretty_json)
            self.txt_config.setHtml(f"{diff_text}<pre style='color:#d4d4d4; font-family:Consolas,monospace; font-size:12px;'>{escaped_json}</pre>")
        except Exception as e:
            self.txt_config.setText(config_str)

    def on_row_double_clicked(self, item):
        r = item.row()
        path = self.btn_open_folder.property("target_path")
        if path and Path(path).exists():
            open_folder(path)

    def on_wrong_image_double_clicked(self, item):
        clicked_img_name = item.text()
        workspace_dir = self.db_manager.db_path.parent
        eval_runs_dir = workspace_dir / "runs" / "eval"
        
        valid_paths = []
        target_idx = 0
        
        run_dir = None
        selected = self.table.selectedItems()
        if selected:
            r = selected[0].row()
            row_data = self.current_rows[r]
            try:
                cfg = json.loads(row_data[8])
                base_run_name = cfg.get("tab3", {}).get("run_name", "check01")
                if "Tab 4" in row_data[3]:
                    run_dir = eval_runs_dir / (base_run_name + "_final_eval")
                else:
                    run_dir = eval_runs_dir / base_run_name
            except: pass

        for i in range(self.list_wrong_imgs.count()):
            name = self.list_wrong_imgs.item(i).text()
            p = run_dir / name if run_dir else None
            
            if not p or not p.exists():
                found = list(eval_runs_dir.glob(f"*/{name}"))
                if found: p = found[0]
                
            if p and p.exists():
                valid_paths.append(str(p))
                if name == clicked_img_name:
                    target_idx = len(valid_paths) - 1
                    
        if valid_paths:
            ImagePreviewDialog(valid_paths, target_idx, self).exec_()
        else:
            QMessageBox.warning(self, "이미지 찾기 실패", f"'{clicked_img_name}' 이미지를 찾을 수 없습니다.\n평가 폴더(runs/eval)에서 삭제되거나 다른 곳으로 이동되었을 수 있습니다.")

    def open_current_folder(self):
        path = self.btn_open_folder.property("target_path")
        if path: open_folder(path)

    def clear_details(self):
        self.lbl_detail_title.setText("<b>[상세 정보]</b> 목록에서 항목을 선택하세요.")
        self.btn_open_folder.setEnabled(False)
        self.txt_config.clear()
        if self.tab_type == 'eval':
            self.list_wrong_imgs.clear()

class LogViewerDialog(QDialog):
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 프로젝트 통합 히스토리 (고급 검색/필터 적용)")
        self.resize(1600, 800)
        self.db_manager = db_manager
        
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        self.train_tab = LogTabWidget('train', self.db_manager, self)
        self.tabs.addTab(self.train_tab, "🏋️ 학습 기록")
        
        self.eval_tab = LogTabWidget('eval', self.db_manager, self)
        self.tabs.addTab(self.eval_tab, "🔍 평가 및 오답 기록")
        
        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.accept)
        btn_close.setMinimumHeight(40)
        layout.addWidget(btn_close)

class TuneHistoryDialog(QDialog):
    def __init__(self, history_csv_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📈 Auto ML 하이퍼파라미터 튜닝 기록")
        self.resize(1000, 700)
        layout = QVBoxLayout(self)

        try:
            df = pd.read_csv(history_csv_path)
            
            self.canvas = FigureCanvas(plt.Figure(figsize=(8, 4)))
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)
            
            ax = self.canvas.figure.add_subplot(111)
            sns.lineplot(data=df, x='generation', y='fitness', marker='o', ax=ax, label='Fitness (잠재력 점수)', color='#e11d48', linewidth=2)
            
            best_row = df.loc[df['fitness'].idxmax()]
            ax.annotate(f"Best: {best_row['fitness']:.2f}", 
                        xy=(best_row['generation'], best_row['fitness']), 
                        xytext=(best_row['generation'], best_row['fitness'] + 1.5),
                        arrowprops=dict(facecolor='#111827', shrink=0.05, width=1.5, headwidth=6),
                        fontsize=10, fontweight='bold')
            
            ax.set_title("세대별 하이퍼파라미터 잠재력(Fitness) 진화 추이", fontdict={'weight': 'bold', 'size': 12})
            ax.set_xlabel("세대 (Generation)")
            ax.set_ylabel("Fitness Score")
            ax.grid(True, linestyle='--', alpha=0.7)
            self.canvas.draw()
            
            layout.addWidget(QLabel("<b>🏆 상위 5개 세대 파라미터 조합 (Top 5)</b>"))
            table = QTableWidget()
            top_df = df.sort_values(by='fitness', ascending=False).head(5)
            
            table.setRowCount(len(top_df))
            table.setColumnCount(len(df.columns))
            table.setHorizontalHeaderLabels(df.columns)
            table.setAlternatingRowColors(True)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            
            for r_idx, (_, row) in enumerate(top_df.iterrows()):
                for c_idx, col_name in enumerate(df.columns):
                    val = row[col_name]
                    if isinstance(val, float): val = f"{val:.4f}"
                    item = QTableWidgetItem(str(val))
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(r_idx, c_idx, item)
            
            table.resizeColumnsToContents()
            layout.addWidget(table)
            
            btn_close = QPushButton("닫기")
            btn_close.setMinimumHeight(40)
            btn_close.clicked.connect(self.accept)
            layout.addWidget(btn_close)
            
        except Exception as e:
            layout.addWidget(QLabel(f"기록을 불러오는 중 오류 발생:\n{traceback.format_exc()}"))


# ==========================================
# Main App
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("YOLO Training Pipeline (PyQt5) - AutoML + Graceful Stop + Embed Webhooks"); self.resize(1400, 900)
        logger.info(f"YOLO Training Pipeline 애플리케이션 시작 (OS: {platform.system()}, GPU: {torch.cuda.is_available()})")
        self.base_dir = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
        default_proj_path = self.base_dir / "MyProject"; self.config_manager = ConfigManager(str(default_proj_path)); self.config_builder = ConfigBuilder()
        self.log_db = LogDatabase(self.base_dir / "MyProject" / "workspace" / "training_history.db"); self.training_process = None; self.init_ui()

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status_animation)
        self.status_animation_frame = 0
        self.base_status_msg = ""

    def update_status_animation(self):
        frames = ["⏳", "⌛"]
        frame = frames[self.status_animation_frame % len(frames)]
        dots = "." * ((self.status_animation_frame % 3) + 1)
        self.status_animation_frame += 1
        pid_info = f"[PID: {self.training_process.pid}] " if self.training_process else ""
        self.statusBar().showMessage(f"{frame} {pid_info}{self.base_status_msg}{dots}")

    def start_dynamic_status(self, msg):
        self.base_status_msg = msg
        self.status_animation_frame = 0
        self.status_timer.start(500)

    def stop_dynamic_status(self, msg=""):
        self.status_timer.stop()
        self.statusBar().showMessage(msg)

    def validate_paths(self, **paths):
        for name, path_obj in paths.items():
            if not path_obj or str(path_obj).strip() in ("", "."):
                return False, f"[{name}] 경로가 입력되지 않았습니다."
                
            path = Path(path_obj).resolve() 
            if not path.exists():
                return False, f"[{name}] 경로를 찾을 수 없습니다:\n{path}"
            if not os.access(path, os.R_OK):
                return False, f"[{name}] 읽기 권한이 없습니다:\n{path}"
            if "work" in name.lower() or "proc" in name.lower() or "output" in name.lower():
                if not os.access(path, os.W_OK):
                    return False, f"[{name}] 쓰기 권한이 없습니다:\n{path}"
                    
            if name.endswith('_file') or name.endswith('.pt'):
                if not path.is_file():
                    return False, f"[{name}] 파일이어야 합니다:\n{path}"
                if path.stat().st_size > 5 * 1024**3:
                    return False, f"[{name}] 파일 용량이 너무 큽니다 (5GB 제한):\n{path}"
                    
            elif name.endswith('_dir') or "ds" in name.lower() or "root" in name.lower():
                if not path.is_dir():
                    return False, f"[{name}] 폴더(디렉토리)여야 합니다:\n{path}"
                    
        return True, None

    def closeEvent(self, event):
        logger.info("애플리케이션 종료 프로세스 시작")
        
        # 1. 중복 알림 방지: 모니터링 스레드 시그널 연결 해제
        # 워커가 종료되면서 던지는 시그널이 closeEvent의 동기 웹훅과 겹치지 않게 합니다.
        if hasattr(self, 'monitor_thread') and self.monitor_thread:
            try:
                self.monitor_thread.finished_ok.disconnect()
                self.monitor_thread.error.disconnect()
                logger.debug("모니터링 스레드 시그널 연결 해제 (중복 알림 방지)")
            except:
                pass

        # 2. 현재 백그라운드 작업(프로세스) 상태 확인
        is_busy = (self.training_process and self.training_process.is_alive())
        
        # 3. 작업 중일 경우 '작업 취소' 웹훅 우선 전송 (동기 방식)
        if is_busy and self.webhook_url and self.noti_flags.get("task"):
            # 현재 어떤 탭의 작업이 수행 중인지 버튼 활성화 상태로 유추
            task_name = "TASK"
            if not self.t2_btn_run.isEnabled(): 
                task_name = "K-FOLD TRAIN / AUTO ML"
            elif not self.t4_btn_retrain.isEnabled(): 
                task_name = "HARD RETRAIN"
            elif not self.t1_btn_run.isEnabled():
                task_name = "DATA PREPROCESS"
            elif not self.t3_btn_run.isEnabled():
                task_name = "EVALUATION"
            elif not self.t5_btn_run.isEnabled():
                task_name = "DISTANCE MEASURE"

            send_discord_webhook(
                webhook_url=self.webhook_url,
                title=f"🛑 [작업 중단] {task_name}",
                description="사용자가 프로그램을 종료하여 진행 중인 작업이 즉시 중단되었습니다.",
                color=0xf39c12, # 주황색
                sync=True # 전송이 완료될 때까지 종료를 잠시 유보 (핵심)
            )

        # 4. 프로그램 전체 종료 알림 웹훅 전송 (동기 방식)
        if self.webhook_url:
            status_text = " (작업 중 종료)" if is_busy else " (정상 종료)"
            send_discord_webhook(
                webhook_url=self.webhook_url,
                title="⏹️ [프로그램 종료]",
                description=f"YOLO 파이프라인 관리 도구가 종료되었습니다.{status_text}",
                color=0x2c3e50, # 남색
                sync=True
            )

        # 5. 백그라운드 프로세스(워커) 안전 종료 시도
        if is_busy:
            self.stop_training() # stop_event.set() 호출
            self.training_process.join(timeout=1.5) # 워커가 정리될 시간을 줌
            if self.training_process.is_alive():
                logger.warning("워커 프로세스가 응답하지 않아 강제 종료(Terminate)합니다.")
                self.training_process.terminate()

        # 6. 기타 UI 스레드(QThread) 정리
        for t_name in ["t1_thread", "t3_thread", "t4_thread", "t5_thread"]:
            if hasattr(self, t_name):
                th = getattr(self, t_name)
                if th and th.isRunning():
                    th.quit()
                    th.wait(1000)

        # 7. QSettings에 현재 설정 저장 및 종료 수락
        self.settings.setValue("webhook_url", self.webhook_url)
        logger.info("애플리케이션 종료 완료")
        event.accept()

    def sync_base_paths(self, new_path):
        if hasattr(self, 't6_img_dir'): self.t6_img_dir.line_edit.setText(str(Path(new_path) / "image"))
        if hasattr(self, 't6_lbl_dir'): self.t6_lbl_dir.line_edit.setText(str(Path(new_path) / "data"))

    def sync_proc_paths(self, new_path):
        if hasattr(self, 't3_img'): self.t3_img.line_edit.setText(str(Path(new_path) / "images"))
        if hasattr(self, 't3_lbl'): self.t3_lbl.line_edit.setText(str(Path(new_path) / "labels"))
        if hasattr(self, 't4_orig'): self.t4_orig.line_edit.setText(str(Path(new_path) / "labels"))
        if hasattr(self, 't5_img'): self.t5_img.line_edit.setText(str(Path(new_path) / "images"))

    def sync_project_root(self, new_path):
        proj_dir = Path(new_path); self.w_base_ds.line_edit.setText(str(proj_dir / "dataset")); self.w_proc_ds.line_edit.setText(str(proj_dir / "processed_dataset")); self.w_work_ds.line_edit.setText(str(proj_dir / "workspace"))
        self.config_manager.update_workspace_path(str(proj_dir / "workspace")); self.log_db = LogDatabase(proj_dir / "workspace" / "training_history.db")
        logger.debug(f"프로젝트 루트 동기화 완료: {new_path}")

    def sync_work_paths(self, new_path):
        if hasattr(self, 't3_model'): self.t3_model.line_edit.setText(str(Path(new_path) / "kfold" / "best_model.pt"))

    def parse_and_validate_class_map(self, text):
        cmap = {}; lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if not lines: return False, "클래스명을 하나 이상 입력해주세요."
        for idx, line in enumerate(lines):
            if not re.match(r'^([a-zA-Z0-9가-힣_-]+)$', line): return False, f"잘못된 클래스명: '{line}'"
            if line in cmap: return False, f"클래스명 '{line}' 중복 입력"
            cmap[line] = idx
        return True, cmap

    def scan_classes_from_data(self):
        logger.info("데이터셋에서 클래스 추출 스캔 시작")
        lbl_dir = Path(self.w_base_ds.get_path()) / "data"; is_valid, err = self.validate_paths(라벨폴더_dir=lbl_dir)
        if not is_valid: QMessageBox.warning(self, "경로 오류", err); return
        found_classes = set()
        for json_file in lbl_dir.glob("*.json"):
            try:
                for item in json.loads(json_file.read_text(encoding="utf-8")).get("area", []):
                    if isinstance(item, dict) and len(item) == 1: found_classes.add(list(item.keys())[0])
            except Exception: continue
        classes_txt = lbl_dir / "classes.txt"
        if classes_txt.exists():
            for line in classes_txt.read_text(encoding="utf-8").splitlines():
                if line.strip(): found_classes.add(line.strip())
        if found_classes:
            sorted_classes = "\n".join(sorted(list(found_classes))); self.t1_class_map.setPlainText(sorted_classes)
            if hasattr(self, 't6_class_map'): self.t6_class_map.setPlainText(sorted_classes)
            self.statusBar().showMessage(f"🔍 스캔 완료: {len(found_classes)}개의 클래스를 찾았습니다.", 4000)
            logger.info(f"스캔 완료: {len(found_classes)} 클래스 찾음")
        else: QMessageBox.warning(self, "스캔 결과", "데이터에서 클래스 이름을 찾지 못했습니다.")

    def bind_default(self, widget, default_val, label_widget):
        widget.setProperty("default_val", default_val)
        def update_lbl(*args):
            val = None
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)): val = widget.value()
            elif isinstance(widget, QComboBox): val = widget.currentText()
            elif isinstance(widget, QTextEdit): val = widget.toPlainText()
            elif isinstance(widget, QLineEdit): val = widget.text()
            elif isinstance(widget, QCheckBox): val = widget.isChecked()
            is_def = False
            if isinstance(val, float) and isinstance(default_val, float): is_def = math.isclose(val, default_val, abs_tol=1e-5)
            elif isinstance(val, str) and isinstance(default_val, str): is_def = (val.strip() == default_val.strip())
            else: is_def = (val == default_val)
            base_text = label_widget.property("base_text")
            if not base_text: base_text = label_widget.text().split("<span")[0].strip(); label_widget.setProperty("base_text", base_text)
            if is_def: label_widget.setText(f"{base_text} <span style='color:#34d399; font-size:11px; font-weight:normal;'>(기본값)</span>")
            else: label_widget.setText(f"{base_text} <span style='color:#f87171; font-size:11px; font-weight:normal;'>(변경됨)</span>")
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)): widget.valueChanged.connect(update_lbl)
        elif isinstance(widget, QComboBox): widget.currentTextChanged.connect(update_lbl)
        elif isinstance(widget, QTextEdit): widget.textChanged.connect(update_lbl)
        elif isinstance(widget, QLineEdit): widget.textChanged.connect(update_lbl)
        elif isinstance(widget, QCheckBox): widget.stateChanged.connect(update_lbl)
        update_lbl()

    def add_param(self, form_layout, label_text, widget, default_val): lbl = QLabel(label_text); form_layout.addRow(lbl, widget); self.bind_default(widget, default_val, lbl)

    def _create_scroll(self, layout): w = QWidget(); w.setLayout(layout); scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(w); return scroll

    def _apply_table_style(self, table): table.setAlternatingRowColors(True); table.setEditTriggers(QTableWidget.NoEditTriggers); table.setSelectionBehavior(QTableWidget.SelectRows)

    def get_noti_flags(self):
        return getattr(self, "noti_flags", {"error": True, "start": True, "early_stop": True, "fold": True, "tune": True, "epoch": True, "epoch_interval": 100, "task": True})

    def show_webhook_settings(self):
        logger.debug("웹훅 설정 다이얼로그 오픈")
        dialog = WebhookSettingsDialog(self.get_noti_flags(), self.webhook_url, self)
        if dialog.exec_() == QDialog.Accepted:
            self.noti_flags = dialog.get_flags(); self.webhook_url = dialog.get_url()
            self.statusBar().showMessage("⚙️ 알림 설정 및 웹훅 URL이 업데이트되었습니다.", 3000)

    def reset_tab_defaults(self, tab_widget, tab_name):
        if QMessageBox.question(self, '초기화 확인', f'{tab_name} 탭의 파라미터를 기본값으로 초기화하시겠습니까?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            logger.info(f"[{tab_name}] 탭 설정 초기화")
            for widget in tab_widget.findChildren(QWidget):
                default_val = widget.property("default_val")
                if default_val is not None:
                    if isinstance(widget, (QSpinBox, QDoubleSpinBox)): widget.setValue(default_val)
                    elif isinstance(widget, QComboBox): widget.setCurrentText(default_val)
                    elif isinstance(widget, QTextEdit): widget.setText(default_val)
                    elif isinstance(widget, QLineEdit): widget.setText(default_val)
                    elif isinstance(widget, QCheckBox): widget.setChecked(default_val)
            if hasattr(self, 't1_class_map') and self.t1_class_map in tab_widget.findChildren(QWidget): self.t1_class_map.setPlainText("OK\nNG")
            if hasattr(self, 't6_class_map') and self.t6_class_map in tab_widget.findChildren(QWidget): self.t6_class_map.setPlainText("OK\nNG")
            self.statusBar().showMessage(f"🔄 {tab_name} 탭이 기본값으로 초기화되었습니다.", 3000)

    def init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget); main_layout = QVBoxLayout(main_widget)
        g_group = QGroupBox("📁 전역 설정 및 시스템 상태"); g_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed); g_layout = QHBoxLayout(); g_layout.setContentsMargins(10, 5, 10, 5)
        path_layout = QVBoxLayout(); path_layout.setSpacing(2)
        self.w_proj_root = PathInputWidget("🌟 프로젝트 루트", True, str(self.base_dir / "MyProject")); self.w_proj_root.line_edit.setStyleSheet("background-color: #fffbeb; font-weight: bold; color: #92400e;")
        default_proj = Path(self.w_proj_root.get_path()); self.w_base_ds = PathInputWidget("원본 데이터(dataset)", True, str(default_proj / "dataset")); self.w_proc_ds = PathInputWidget("처리 폴더(processed)", True, str(default_proj / "processed_dataset")); self.w_work_ds = PathInputWidget("워크스페이스(workspace)", True, str(default_proj / "workspace"))
        
        self.settings = QSettings("MyVisionProject", "YoloTrainerApp")
        self.webhook_url = self.settings.value("webhook_url", "")
        self.noti_flags = {"error": True, "start": True, "early_stop": True, "fold": True, "tune": True, "epoch": True, "epoch_interval": 100, "task": True}
        
        self.btn_webhook_settings = QPushButton("🔔 디스코드 알림 설정")
        self.btn_webhook_settings.setStyleSheet("background-color: #fce7f3; border: 1px solid #fbcfe8; font-weight: bold; color: #9d174d; padding: 5px 15px; border-radius: 4px;")
        self.btn_webhook_settings.clicked.connect(self.show_webhook_settings)

        webhook_h_layout = QHBoxLayout(); webhook_h_layout.setSpacing(10); webhook_h_layout.addStretch(); webhook_h_layout.addWidget(self.btn_webhook_settings)

        path_layout.addWidget(self.w_proj_root); path_layout.addWidget(self.w_base_ds); path_layout.addWidget(self.w_proc_ds); path_layout.addWidget(self.w_work_ds); path_layout.addSpacing(5); path_layout.addLayout(webhook_h_layout)
        right_layout = QVBoxLayout(); right_layout.setSpacing(2)
        gpu_info = f"🟢 {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)" if torch.cuda.is_available() else "GPU 없음 (CPU 연산)"
        sys_label = QLabel(f"🖥️ <b>OS:</b> {platform.system()} {platform.release()} &nbsp;&nbsp;|&nbsp;&nbsp; 💾 <b>RAM:</b> {psutil.virtual_memory().total / (1024**3):.1f} GB &nbsp;&nbsp;|&nbsp;&nbsp; 🚀 <b>GPU:</b> {gpu_info}"); sys_label.setStyleSheet("color: #4b5563; font-size: 12px;"); right_layout.addWidget(sys_label)
        opt_layout = QHBoxLayout()
        d_glob = ConfigDefaults.GLOBAL
        self.g_model = QComboBox(); self.g_model.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt", "yolo11n.pt", "yolo11s.pt"]); lbl_mod = QLabel("Model:"); self.bind_default(self.g_model, d_glob["model"], lbl_mod)
        self.g_imgsz = QComboBox(); self.g_imgsz.addItems(["320", "416", "512", "640", "768", "1024"]); self.g_imgsz.setCurrentText(d_glob["imgsz"]); lbl_sz = QLabel("imgsz:"); self.bind_default(self.g_imgsz, d_glob["imgsz"], lbl_sz)
        opt_layout.addWidget(lbl_mod); opt_layout.addWidget(self.g_model); opt_layout.addSpacing(20); opt_layout.addWidget(lbl_sz); opt_layout.addWidget(self.g_imgsz); opt_layout.addStretch(); right_layout.addLayout(opt_layout)
        btn_layout1 = QHBoxLayout()
        self.btn_save_config = QPushButton("💾 설정만 저장"); self.btn_save_config.setStyleSheet("background-color: #f3f4f6; border: 1px solid #d1d5db; padding: 5px; border-radius: 4px;"); self.btn_save_config.clicked.connect(self.save_config_dialog)
        self.btn_load_config = QPushButton("📂 설정만 불러오기"); self.btn_load_config.setStyleSheet("background-color: #f3f4f6; border: 1px solid #d1d5db; padding: 5px; border-radius: 4px;"); self.btn_load_config.clicked.connect(self.load_config_dialog)
        self.btn_reset_all = QPushButton("🔄 전체 초기화"); self.btn_reset_all.setStyleSheet("background-color: #fee2e2; border: 1px solid #fca5a5; padding: 5px; border-radius: 4px;"); self.btn_reset_all.clicked.connect(self.reset_all_defaults)
        self.btn_show_logs = QPushButton("📊 학습 이력(DB)"); self.btn_show_logs.setStyleSheet("background-color: #e0e7ff; border: 1px solid #a5b4fc; padding: 5px; border-radius: 4px; font-weight: bold; color: #3730a3;"); self.btn_show_logs.clicked.connect(self.show_log_viewer)
        btn_layout1.addWidget(self.btn_save_config); btn_layout1.addWidget(self.btn_load_config); btn_layout1.addWidget(self.btn_reset_all); btn_layout1.addWidget(self.btn_show_logs)
        btn_layout2 = QHBoxLayout()
        self.btn_export_proj = QPushButton("📦 전체 백업 (내보내기)"); self.btn_export_proj.setStyleSheet("background-color: #dbeafe; border: 1px solid #93c5fd; padding: 5px; border-radius: 4px; font-weight: bold; color: #1e3a8a;"); self.btn_export_proj.clicked.connect(self.export_project_dialog)
        self.btn_import_proj = QPushButton("📥 전체 복구 (불러오기)"); self.btn_import_proj.setStyleSheet("background-color: #dbeafe; border: 1px solid #93c5fd; padding: 5px; border-radius: 4px; font-weight: bold; color: #1e3a8a;"); self.btn_import_proj.clicked.connect(self.import_project_dialog)
        btn_layout2.addWidget(self.btn_export_proj); btn_layout2.addWidget(self.btn_import_proj); right_layout.addLayout(btn_layout1); right_layout.addLayout(btn_layout2)
        g_layout.addLayout(path_layout, 5); g_layout.addSpacing(20); g_layout.addLayout(right_layout, 5); g_group.setLayout(g_layout); main_layout.addWidget(g_group)
        self.tabs = QTabWidget(); main_layout.addWidget(self.tabs); self.setup_tab6(); self.setup_tab1(); self.setup_tab2(); self.setup_tab3(); self.setup_tab4(); self.setup_tab5()
        self.w_base_ds.line_edit.textChanged.connect(self.sync_base_paths); self.w_proc_ds.line_edit.textChanged.connect(self.sync_proc_paths); self.w_proj_root.line_edit.textChanged.connect(self.sync_project_root); self.w_work_ds.line_edit.textChanged.connect(self.sync_work_paths)

    def show_log_viewer(self): 
        logger.debug("DB 로그 뷰어 오픈")
        LogViewerDialog(self.log_db, self).exec_()

    def reset_all_defaults(self):
        if QMessageBox.question(self, '전체 초기화 확인', '전체 탭의 모든 파라미터를 기본값으로 되돌리시겠습니까?\n(경로 설정은 유지됩니다)', QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            logger.info("사용자 요청에 의해 전체 파라미터 초기화 진행")
            for widget in self.findChildren(QWidget):
                default_val = widget.property("default_val")
                if default_val is not None:
                    if isinstance(widget, (QSpinBox, QDoubleSpinBox)): widget.setValue(default_val)
                    elif isinstance(widget, QComboBox): widget.setCurrentText(default_val)
                    elif isinstance(widget, QTextEdit): widget.setText(default_val)
                    elif isinstance(widget, QLineEdit): widget.setText(default_val)
                    elif isinstance(widget, QCheckBox): widget.setChecked(default_val)
            if hasattr(self, 't1_class_map'): self.t1_class_map.setPlainText("OK\nNG")
            if hasattr(self, 't6_class_map'): self.t6_class_map.setPlainText("OK\nNG")
            self.statusBar().showMessage("🔄 모든 설정이 기본값으로 초기화되었습니다.", 3000)

    def save_config_dialog(self):
        text, ok = QInputDialog.getText(self, '설정 저장', '저장할 설정의 이름을 입력하세요:\n(빈칸으로 두면 날짜/시간으로 자동 생성됩니다)')
        if ok:
            success, result = self.config_manager.save_config(self.config_builder.build(self), text.strip())
            if success: self.statusBar().showMessage(f"💾 설정이 성공적으로 저장되었습니다: {Path(result).name}", 5000); QMessageBox.information(self, "저장 완료", f"설정이 저장되었습니다.\n{result}")
            else: QMessageBox.critical(self, "저장 실패", f"설정 저장 중 오류가 발생했습니다:\n{result}")

    def load_config_dialog(self):
        configs = self.config_manager.get_all_configs()
        if not configs: QMessageBox.information(self, "알림", "저장된 설정 파일이 없습니다."); return
        dialog = QDialog(self); dialog.setWindowTitle("설정 불러오기"); dialog.resize(400, 300); layout = QVBoxLayout(dialog); list_widget = QListWidget()
        for f in configs: list_widget.addItem(f.name)
        layout.addWidget(QLabel("불러올 설정을 선택하세요:")); layout.addWidget(list_widget); btn_layout = QHBoxLayout()
        btn_load = QPushButton("불러오기"); btn_delete = QPushButton("삭제"); btn_cancel = QPushButton("취소")
        btn_layout.addWidget(btn_load); btn_layout.addWidget(btn_delete); btn_layout.addWidget(btn_cancel); layout.addLayout(btn_layout)
        def load_selected():
            if list_widget.currentItem():
                file_name = list_widget.currentItem().text(); success, data = self.config_manager.load_config(self.config_manager.config_dir / file_name)
                if success: self.apply_loaded_config(data); self.statusBar().showMessage(f"📂 설정을 불러왔습니다: {file_name}", 5000); dialog.accept()
                else: QMessageBox.critical(self, "오류", f"파일을 읽는 중 오류가 발생했습니다:\n{data}")
        def delete_selected():
            if list_widget.currentItem():
                file_name = list_widget.currentItem().text(); file_path = self.config_manager.config_dir / file_name
                if QMessageBox.question(dialog, '삭제 확인', f"'{file_name}' 설정을 삭제하시겠습니까?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes: 
                    file_path.unlink(missing_ok=True); list_widget.takeItem(list_widget.currentRow()); logger.info(f"설정 파일 삭제됨: {file_name}")
        btn_load.clicked.connect(load_selected); btn_delete.clicked.connect(delete_selected); btn_cancel.clicked.connect(dialog.reject); list_widget.itemDoubleClicked.connect(load_selected); dialog.exec_()

    def apply_loaded_config(self, c):
        logger.debug("불러온 Config를 UI에 적용 중")
        g = c.get("global", {})
        if "proj_root" in g: self.w_proj_root.line_edit.setText(g["proj_root"])
        if "base_ds" in g: self.w_base_ds.line_edit.setText(g["base_ds"])
        if "proc_ds" in g: self.w_proc_ds.line_edit.setText(g["proc_ds"])
        if "work_ds" in g: self.w_work_ds.line_edit.setText(g["work_ds"])
        if "webhook_url" in g and g["webhook_url"].strip(): self.webhook_url = g["webhook_url"]
        if "noti_flags" in g: self.noti_flags = g["noti_flags"]
        if "model" in g: self.g_model.setCurrentText(g["model"])
        if "imgsz" in g: self.g_imgsz.setCurrentText(g["imgsz"])
        t1 = c.get("tab1", {})
        if "auto_crop" in t1: self.t1_auto_crop.setChecked(t1["auto_crop"])
        if "margin" in t1: self.t1_margin.setValue(t1["margin"])
        if "mx" in t1: self.t1_mx.setValue(t1["mx"])
        if "my" in t1: self.t1_my.setValue(t1["my"])
        if "mw" in t1: self.t1_mw.setValue(t1["mw"])
        if "mh" in t1: self.t1_mh.setValue(t1["mh"])
        if "class_map" in t1: self.t1_class_map.setText(t1["class_map"])
        if "clean" in t1: self.t1_clean.setChecked(t1["clean"])
        if "exif" in t1: self.t1_exif.setChecked(t1["exif"])
        t2 = c.get("tab2", {})
        if "epochs" in t2: self.t2_epochs.setValue(t2["epochs"])
        if "batch" in t2: self.t2_batch.setValue(t2["batch"])
        if "workers" in t2: self.t2_workers.setValue(t2["workers"])
        if "patience" in t2: self.t2_patience.setValue(t2["patience"])
        if "seed" in t2: self.t2_seed.setValue(t2["seed"])
        if "folds" in t2: self.t2_folds.setValue(t2["folds"])
        if "test_split" in t2: self.t2_test_split.setValue(t2["test_split"])
        if "lcls" in t2: self.t2_lcls.setValue(t2["lcls"])
        if "lbox" in t2: self.t2_lbox.setValue(t2["lbox"])
        if "ldfl" in t2: self.t2_ldfl.setValue(t2["ldfl"])
        if "ah" in t2: self.t2_ah.setValue(t2["ah"])
        if "as" in t2: self.t2_as.setValue(t2["as"])
        if "av" in t2: self.t2_av.setValue(t2["av"])
        if "adeg" in t2: self.t2_adeg.setValue(t2["adeg"])
        if "atrans" in t2: self.t2_atrans.setValue(t2["atrans"])
        if "ascale" in t2: self.t2_ascale.setValue(t2["ascale"])
        if "ashear" in t2: self.t2_ashear.setValue(t2["ashear"])
        if "afud" in t2: self.t2_afud.setValue(t2["afud"])
        if "aflr" in t2: self.t2_aflr.setValue(t2["aflr"])
        if "amos" in t2: self.t2_amos.setValue(t2["amos"])
        if "amix" in t2: self.t2_amix.setValue(t2["amix"])
        if "acp" in t2: self.t2_acp.setValue(t2["acp"])
        t3 = c.get("tab3", {})
        if "conf" in t3: self.t3_conf.setValue(t3["conf"])
        if "iou" in t3: self.t3_iou.setValue(t3["iou"])
        if "match_iou" in t3: self.t3_match_iou.setValue(t3["match_iou"])
        if "max_det" in t3: self.t3_max_det.setValue(t3["max_det"])
        if "run_name" in t3: self.t3_run_name.setText(t3["run_name"])
        if "agnostic" in t3: self.t3_agnostic.setChecked(t3["agnostic"])
        if "save_rel" in t3: self.t3_save_rel.setChecked(t3["save_rel"])
        t4 = c.get("tab4", {})
        if "epochs" in t4: self.t4_epochs.setValue(t4["epochs"])
        if "batch" in t4: self.t4_batch.setValue(t4["batch"])
        if "run" in t4: self.t4_run.setText(t4["run"])
        if "lcls" in t4: self.t4_lcls.setValue(t4["lcls"])
        if "lbox" in t4: self.t4_lbox.setValue(t4["lbox"])
        if "ah" in t4: self.t4_ah.setValue(t4["ah"])
        if "as" in t4: self.t4_as.setValue(t4["as"])
        if "av" in t4: self.t4_av.setValue(t4["av"])
        if "afud" in t4: self.t4_afud.setValue(t4["afud"])
        if "aflr" in t4: self.t4_aflr.setValue(t4["aflr"])
        if "amos" in t4: self.t4_amos.setValue(t4["amos"])
        if "amix" in t4: self.t4_amix.setValue(t4["amix"])
        if "acp" in t4: self.t4_acp.setValue(t4["acp"])
        if "eval_conf" in t4: self.t4_conf.setValue(t4["eval_conf"])
        if "eval_iou" in t4: self.t4_iou.setValue(t4["eval_iou"])
        if "eval_match" in t4: self.t4_match_iou.setValue(t4["eval_match"])
        if "eval_max" in t4: self.t4_max_det.setValue(t4["eval_max"])
        if "eval_agnostic" in t4: self.t4_agnostic.setChecked(t4["eval_agnostic"])
        t5 = c.get("tab5", {})
        if "method" in t5: self.t5_method.setCurrentText(t5["method"])
        if "conf" in t5: self.t5_conf.setValue(t5["conf"])
        if "iou" in t5: self.t5_iou.setValue(t5["iou"])
        if "max_det" in t5: self.t5_max_det.setValue(t5["max_det"])
        if "agnostic" in t5: self.t5_agnostic.setChecked(t5["agnostic"])
        if "knn_n" in t5: self.t5_knn_n.setValue(t5["knn_n"])
        if "edge_thr" in t5: self.t5_edge_thr.setText(t5["edge_thr"])
        if "skip_ng" in t5: self.t5_skip_ng.setChecked(t5["skip_ng"])
        if "drop_odd" in t5: self.t5_drop_odd.setChecked(t5["drop_odd"])
        if "color1" in t5: self.t5_color1.setCurrentText(t5["color1"])
        if "color2" in t5: self.t5_color2.setCurrentText(t5["color2"])
        t6 = c.get("tab6", {})
        if "class_map" in t6: self.t6_class_map.setText(t6["class_map"])
        if "color" in t6: self.t6_color_combo.setCurrentText(t6["color"])
        if "auto_thr" in t6: self.t6_auto_thr.setValue(t6["auto_thr"])
        if "auto_nms" in t6: self.t6_auto_nms.setValue(t6["auto_nms"])

    def start_training_process(self, worker_func, args):
        logger.info(f"멀티프로세싱 워커 시작. 대상 함수: {worker_func.__name__}")
        
        self.stop_event = multiprocessing.Event()
        args["stop_event"] = self.stop_event
        
        self.train_queue = multiprocessing.Queue()
        self.training_process = multiprocessing.Process(target=worker_func, args=(args, self.train_queue))
        self.training_process.start()
        
        self.monitor_thread = ProcessMonitorThread(self.train_queue, self.training_process)
        self.monitor_thread.finished_ok.connect(self.on_training_finished)
        self.monitor_thread.error.connect(self.on_training_fatal_error)
        self.monitor_thread.start()
        
        webhook_url = self.webhook_url
        if webhook_url and self.noti_flags.get("start"): 
            send_discord_webhook(
                webhook_url=webhook_url, 
                title="▶️ [작업 시작]", 
                description="새로운 백그라운드 작업이 시작되었습니다.",
                color=0x9b59b6
            )
        
        self.btn_webhook_settings.setEnabled(False)
        self.t2_btn_run.setEnabled(False); self.t2_btn_tune.setEnabled(False); self.t4_btn_retrain.setEnabled(False); self.t4_btn_auto_thr.setEnabled(False); self.t3_btn_auto_thr.setEnabled(False); self.t2_scroll.setEnabled(False); self.t4_scroll.setEnabled(False); self.g_model.setEnabled(False); self.g_imgsz.setEnabled(False)
        
        self.t2_btn_stop.setText("🛑 안전 종료(Graceful Stop)")
        self.t2_btn_stop.setEnabled(True)
        
        self.start_dynamic_status("백그라운드 작업 진행 중")

    def stop_training(self):
        logger.warning("사용자에 의한 안전한 프로세스 종료(Graceful Stop) 요청")
        if self.training_process and self.training_process.is_alive():
            if hasattr(self, 'stop_event'):
                self.stop_event.set()
                self.statusBar().showMessage("🛑 진행 중인 작업을 안전하게 마무리하고 종료합니다. 잠시만 기다려주십시오...")
                self.t2_btn_stop.setEnabled(False)
                self.t2_btn_stop.setText("종료 처리 중...")

    def on_training_finished(self, res):
        logger.info(f"워커 프로세스 완료. 결과: Success={res.get('success')}, Task={res.get('task')}")
        self._restore_training_ui()
        
        self.stop_dynamic_status("✅ 프로세스 완료")
        
        task_name = str(res.get('task', 'UNKNOWN')).upper()
        
        # 사용자가 안전 종료 버튼을 눌렀는지 확인
        is_stopped_early = hasattr(self, 'stop_event') and self.stop_event.is_set()

        if res.get("success"):
            if is_stopped_early:
                # 안전 종료 시의 웹훅 전송 (주황색)
                if self.webhook_url and self.noti_flags.get("task"):
                    send_discord_webhook(
                        webhook_url=self.webhook_url,
                        title=f"🛑 [작업 중단] {task_name}",
                        description="사용자 요청에 의해 작업이 안전하게 종료(Graceful Stop)되었습니다.\n*(중단 전까지 학습된 가중치와 결과 데이터는 정상 보존됩니다.)*",
                        color=0xf39c12
                    )
            else:
                # 일반적인 완료 시의 웹훅 전송 (초록색)
                if self.webhook_url and self.noti_flags.get("task"):
                    send_discord_webhook(
                        webhook_url=self.webhook_url,
                        title=f"✅ [작업 완료] {task_name}",
                        description="작업이 성공적으로 완료되었습니다.",
                        color=0x2ecc71
                    )

            task = res.get("task"); current_config = self.config_builder.build(self)
            if task == "tune":
                bp = res.get("best_params", {})
                if 'box' in bp: self.t2_lbox.setValue(float(bp['box']))
                if 'cls' in bp: self.t2_lcls.setValue(float(bp['cls']))
                if 'dfl' in bp: self.t2_ldfl.setValue(float(bp['dfl']))
                if 'hsv_h' in bp: self.t2_ah.setValue(float(bp['hsv_h']))
                if 'hsv_s' in bp: self.t2_as.setValue(float(bp['hsv_s']))
                if 'hsv_v' in bp: self.t2_av.setValue(float(bp['hsv_v']))
                if 'degrees' in bp: self.t2_adeg.setValue(float(bp['degrees']))
                if 'translate' in bp: self.t2_atrans.setValue(float(bp['translate']))
                if 'scale' in bp: self.t2_ascale.setValue(float(bp['scale']))
                if 'shear' in bp: self.t2_ashear.setValue(float(bp['shear']))
                if 'flipud' in bp: self.t2_afud.setValue(float(bp['flipud']))
                if 'fliplr' in bp: self.t2_aflr.setValue(float(bp['fliplr']))
                if 'mosaic' in bp: self.t2_amos.setValue(float(bp['mosaic']))
                if 'mixup' in bp: self.t2_amix.setValue(float(bp['mixup']))
                if 'copy_paste' in bp: self.t2_acp.setValue(float(bp['copy_paste']))
                
                if "history" in res:
                    try:
                        history_df = pd.DataFrame(res["history"])
                        history_path = Path(self.w_work_ds.get_path()) / "runs" / "tune_custom" / "tune_history.csv"
                        history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
                        logger.info(f"AutoML 탐색 기록 CSV 저장 완료: {history_path}")
                    except Exception as e:
                        logger.error(f"AutoML 탐색 기록 CSV 저장 중 오류: {e}")

                QMessageBox.information(self, "Auto ML 완료", res["msg"])
            elif task == "threshold":
                bc, bi = res["best_conf"], res["best_iou"]
                self.t3_conf.setValue(bc); self.t3_iou.setValue(bi); self.t4_conf.setValue(bc); self.t4_iou.setValue(bi); self.t5_conf.setValue(bc); self.t5_iou.setValue(bi)
                QMessageBox.information(self, "스레숄드 탐색 완료", res["msg"])
            elif task == "train":
                self.show_kfold_metrics_dialog(res["metrics_summary"], res["msg"], res.get("best_fold", ""))
                avg_map = next((summary.get("mAP50-95", 0.0) for summary in res["metrics_summary"] if summary["Fold"] == "Average"), 0.0)
                original_path = res.get("original_model_path", res.get("best_model", "경로 없음"))
                self.log_db.insert_log(task_type="K-Fold Train", model_name=self.g_model.currentText(), epochs=self.t2_epochs.value(), batch=self.t2_batch.value(), best_map=avg_map, save_dir=original_path, config_data=current_config)
                if res.get("best_model"): self.t3_model.line_edit.setText(res["best_model"])
            elif task == "retrain":
                QMessageBox.information(self, "완료", res["msg"])
                self.log_db.insert_log(task_type="Hard Retrain", model_name=Path(self.t4_base.get_path()).name, epochs=self.t4_epochs.value(), batch=self.t4_batch.value(), best_map=-1.0, save_dir=res.get("model_path", "경로 없음"), config_data=current_config)
                if res.get("model_path"): 
                    self.t4_eval_model_display.setText(res["model_path"]); self.t4_btn_eval.setEnabled(True); self.statusBar().showMessage(f"✅ 재학습 모델 준비 완료: {Path(res['model_path']).name}")
        else: 
            # 에러 또는 취소 처리 로직 분기
            error_msg = res.get("error", "알 수 없는 오류")
            
            # 1. 사용자가 의도적으로 중단/취소한 경우 (에러가 아님)
            if "사용자에 의해" in error_msg or "취소" in error_msg:
                if self.webhook_url and self.noti_flags.get("task"):
                    send_discord_webhook(self.webhook_url, f"🛑 [작업 취소] {task_name}", "사용자의 요청으로 작업이 안전하게 취소/중단되었습니다.", color=0xf39c12)
                QMessageBox.information(self, "작업 취소 안내", error_msg)
                
            # 2. 진짜 에러가 발생한 경우
            else:
                if self.webhook_url and self.noti_flags.get("task"):
                    send_discord_webhook(self.webhook_url, f"⚠️ [작업 실패] {task_name}", f"작업이 비정상 종료되었습니다.\n상세: {error_msg}", color=0xe74c3c)
                QMessageBox.critical(self, "결과 알림", error_msg)
        
        self.training_process = None

    def on_training_fatal_error(self, error_msg):
        logger.error(f"프로세스 비정상 종료 콜백 수신. 원인: {error_msg}")
        if self.training_process is None: return
        webhook_url = self.webhook_url
        if webhook_url and self.noti_flags.get("error"): 
            send_discord_webhook(
                webhook_url=webhook_url,
                title="❌ [프로세스 비정상 종료]",
                description=f"상세 원인:\n```{error_msg}```",
                color=0xe74c3c
            )
        self._restore_training_ui(); QMessageBox.critical(self, "비정상 종료", error_msg)
        
        self.stop_dynamic_status("🛑 프로세스가 비정상 종료되었습니다.")
        self.training_process = None

    def on_thread_error(self, task_name, error_msg):
        logger.error(f"[{task_name}] QThread 스레드 내부 예외 발생. 원인: {error_msg}")
        if self.webhook_url and self.noti_flags.get("error"):
            send_discord_webhook(
                webhook_url=self.webhook_url,
                title="❌ [프로세스 에러 발생]",
                description=f"[{task_name}] 작업 중 오류 발생.\n상세 원인:\n```{error_msg}```",
                color=0xe74c3c
            )
        QMessageBox.critical(self, f"{task_name} 오류", error_msg)
        self.statusBar().showMessage(f"🛑 {task_name} 작업이 비정상 종료되었습니다.")

    def _restore_training_ui(self):
        self.btn_webhook_settings.setEnabled(True)
        self.t2_btn_run.setEnabled(True); self.t2_btn_tune.setEnabled(True); self.t4_btn_retrain.setEnabled(True); self.t4_btn_auto_thr.setEnabled(True); self.t3_btn_auto_thr.setEnabled(True); self.t2_scroll.setEnabled(True); self.t4_scroll.setEnabled(True); self.g_model.setEnabled(True); self.g_imgsz.setEnabled(True)
        self.t2_btn_stop.setEnabled(False)
        self.t2_btn_stop.setText("🛑 안전 종료(Graceful Stop)")

    def show_kfold_metrics_dialog(self, metrics_data, title_msg, best_fold):
        logger.debug(f"K-Fold 검증 다이얼로그 오픈 (Best Fold: {best_fold})")
        dialog = QDialog(self); dialog.setWindowTitle("K-Fold 교차 검증 상세 지표"); dialog.resize(650, 350); layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"<b>{title_msg.replace(chr(10), '<br>')}</b><br><span style='color: #ef4444;'>⭐ <b>최우수 모델: Fold {best_fold}</b> (가중치 자동 저장됨)</span>"))
        table = QTableWidget(len(metrics_data), 6); table.setHorizontalHeaderLabels(["Fold", "mAP50", "mAP50-95", "Precision", "Recall", "Fitness"]); self._apply_table_style(table)
        for i in range(table.columnCount()): table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        for row, data in enumerate(metrics_data):
            fold_name = str(data["Fold"])
            table.setItem(row, 0, QTableWidgetItem(f"👑 Fold {fold_name}" if fold_name == str(best_fold) else fold_name)); table.setItem(row, 1, QTableWidgetItem(f"{data['mAP50']:.4f}")); table.setItem(row, 2, QTableWidgetItem(f"{data['mAP50-95']:.4f}")); table.setItem(row, 3, QTableWidgetItem(f"{data['Precision']:.4f}")); table.setItem(row, 4, QTableWidgetItem(f"{data['Recall']:.4f}")); table.setItem(row, 5, QTableWidgetItem(f"{data['Fitness']:.4f}")) 
            for col in range(6): 
                item = table.item(row, col); item.setTextAlignment(Qt.AlignCenter)
                if fold_name == "Average": font = item.font(); font.setBold(True); item.setFont(font); item.setBackground(QColor("#dbeafe")) 
                elif fold_name == str(best_fold): font = item.font(); font.setBold(True); item.setFont(font); item.setBackground(QColor("#fef08a")) 
        layout.addWidget(table); btn = QPushButton("확인"); btn.clicked.connect(dialog.accept); layout.addWidget(btn); dialog.exec_()

    def select_prev_label_image(self):
        curr_row = self.t6_list.currentRow(); 
        if curr_row > 0: self.t6_list.setCurrentRow(curr_row - 1); self.on_label_image_selected(self.t6_list.currentItem())
    def select_next_label_image(self):
        curr_row = self.t6_list.currentRow(); 
        if curr_row < self.t6_list.count() - 1: self.t6_list.setCurrentRow(curr_row + 1); self.on_label_image_selected(self.t6_list.currentItem())

    def load_labeling_images(self):
        logger.info("라벨링용 원본 이미지 폴더 탐색")
        src_dir = QFileDialog.getExistingDirectory(self, "복사해올 원본 이미지 폴더 선택")
        if not src_dir: return
        src_path = Path(src_dir); target_dir = Path(self.t6_img_dir.get_path()); target_dir.mkdir(parents=True, exist_ok=True); valid_exts = {".jpg", ".jpeg", ".png"}
        files_to_copy = [f for f in src_path.iterdir() if f.suffix.lower() in valid_exts]
        if not files_to_copy: QMessageBox.warning(self, "파일 없음", "선택한 폴더에 이미지 파일이 없습니다."); return
        copy_count = 0; QApplication.setOverrideCursor(Qt.WaitCursor) 
        try:
            for f in files_to_copy:
                target_file = target_dir / f.name
                if not target_file.exists(): shutil.copy2(f, target_file); copy_count += 1
            self.t6_list.clear()
            for f in sorted(target_dir.iterdir()):
                if f.suffix.lower() in valid_exts: self.t6_list.addItem(f.name)
        finally: QApplication.restoreOverrideCursor()
        msg = f"작업 폴더로 {copy_count}장의 사진을 복사했습니다. (현재 목록: 총 {self.t6_list.count()}장)"
        self.statusBar().showMessage(msg, 5000); QMessageBox.information(self, "가져오기 완료", msg)

    def on_label_image_selected(self, item):
        img_path = Path(self.t6_img_dir.get_path()) / item.text(); success, cmap_or_error = self.parse_and_validate_class_map(self.t6_class_map.toPlainText())
        if not success: QMessageBox.warning(self, "클래스 매핑 오류", cmap_or_error); return
        class_names = [k for k, v in sorted(cmap_or_error.items(), key=lambda item: item[1])]
        if not self.t6_view.load_image(str(img_path), class_names): QMessageBox.warning(self, "오류", "이미지를 불러올 수 없습니다."); return
        self.t6_label_list.clear(); lbl_dir = Path(self.t6_lbl_dir.get_path()); txt_path = lbl_dir / Path(item.text()).with_suffix(".txt").name
        if txt_path.exists(): self.t6_view.load_existing_labels([line for line in txt_path.read_text(encoding="utf-8").strip().split('\n') if line.strip()])

    def save_current_label(self, yolo_data):
        current_item = self.t6_list.currentItem()
        if not current_item: return
        lbl_dir = Path(self.t6_lbl_dir.get_path()); lbl_dir.mkdir(parents=True, exist_ok=True); txt_path = lbl_dir / Path(current_item.text()).with_suffix(".txt").name
        if yolo_data: txt_path.write_text("\n".join(yolo_data), encoding="utf-8"); current_item.setBackground(QColor("#dcfce7"))
        else:
            if txt_path.exists(): txt_path.unlink()
            current_item.setBackground(QColor("#ffffff"))
        self.statusBar().showMessage(f"✅ 라벨 저장 완료: {txt_path.name}", 2000)

    def send_to_measure_tab(self, model_path, img_path, conf, iou, max_det, agnostic):
        if not model_path or not Path(model_path).exists(): QMessageBox.warning(self, "경로 오류", "유효한 모델 경로가 없습니다."); return
        self.t5_model.line_edit.setText(model_path); self.t5_img.line_edit.setText(img_path); self.t5_conf.setValue(conf); self.t5_iou.setValue(iou); self.t5_max_det.setValue(max_det); self.t5_agnostic.setChecked(agnostic); self.tabs.setCurrentIndex(5); self.statusBar().showMessage("➡️ 현재 모델, 이미지 경로, 평가 설정이 거리 측정 탭으로 복사되었습니다.", 5000)

    def sync_label_ui(self, yolo_data):
        selected_rows = [item.row() for item in self.t6_label_list.selectedIndexes()]
        self.t6_label_list.blockSignals(True); self.t6_label_list.clear()
        for i, line in enumerate(yolo_data):
            parts = line.split()
            if len(parts) == 5:
                class_id = int(parts[0]); class_name = self.t6_view.class_names[class_id] if class_id < len(self.t6_view.class_names) else f"ID {class_id}"
                self.t6_label_list.addItem(f"[{i+1}] {class_name} (X:{float(parts[1]):.3f}, Y:{float(parts[2]):.3f})")
        for row in selected_rows:
            if 0 <= row < self.t6_label_list.count(): self.t6_label_list.item(row).setSelected(True)
        self.t6_label_list.blockSignals(False); self.on_label_selection_changed(); self.save_current_label(yolo_data)

    def on_label_selection_changed(self): self.t6_view.select_boxes([item.row() for item in self.t6_label_list.selectedIndexes()])

    def on_delete_label_clicked(self):
        selected_rows = [item.row() for item in self.t6_label_list.selectedIndexes()]
        if selected_rows: self.t6_view.delete_boxes(selected_rows); self.t6_view.setFocus() 
        else: QMessageBox.warning(self, "선택 오류", "삭제할 라벨을 선택해주세요.")

    def on_change_class_clicked(self):
        selected_rows = [item.row() for item in self.t6_label_list.selectedIndexes()]
        if selected_rows:
            dialog = LabelDialog(self.t6_view.class_names, self)
            if dialog.exec_() == QDialog.Accepted: self.t6_view.update_boxes_class(selected_rows, dialog.get_class_index())
            self.t6_view.setFocus() 
        else: QMessageBox.warning(self, "선택 오류", "클래스를 변경할 라벨을 선택해주세요.")

    def run_auto_labeling(self):
        logger.info("오토 라벨링 기능 수행")
        threshold = self.t6_auto_thr.value(); success, result = self.t6_view.auto_label_similar(threshold, self.t6_auto_nms.value())
        if not success: QMessageBox.warning(self, "오토 라벨링 실패", result)
        else:
            if result == 0: self.statusBar().showMessage("⚠️ 비슷한 객체를 찾지 못했습니다.", 4000)
            else: self.statusBar().showMessage(f"✨ {result}개의 유사한 객체를 자동 라벨링했습니다!", 4000)

    def run_auto_labeling_next(self):
        logger.info("다음 사진 일괄 오토 라벨링 실행")
        if not self.t6_view.save_all_templates(): QMessageBox.warning(self, "안내", "먼저 기준이 될 박스를 그려주세요."); return
        curr_row = self.t6_list.currentRow()
        if curr_row < self.t6_list.count() - 1: self.t6_list.setCurrentRow(curr_row + 1); self.on_label_image_selected(self.t6_list.currentItem())
        else: QMessageBox.information(self, "안내", "마지막 사진입니다."); return
        threshold = self.t6_auto_thr.value(); success, result = self.t6_view.auto_label_from_saved_templates(threshold, self.t6_auto_nms.value())
        if not success: QMessageBox.warning(self, "오토 라벨링 실패", result)
        else:
            if result == 0: self.statusBar().showMessage(f"⚠️ 비슷한 객체를 찾지 못했습니다.", 4000)
            else: self.statusBar().showMessage(f"✨ 총 {result}개의 객체를 찾아 일괄 라벨링했습니다!", 4000); self.sync_label_ui(self.t6_view.get_yolo_format())

    def setup_tab6(self):
        f = QFormLayout(); base_dir = Path(self.w_base_ds.get_path()); self.t6_img_dir = PathInputWidget("작업할 이미지 폴더", True, str(base_dir/"image")); self.t6_lbl_dir = PathInputWidget("라벨 저장 폴더 (.txt)", True, str(base_dir/"data"))
        self.t6_class_map = QTextEdit(); self.t6_class_map.setPlainText("OK\nNG"); self.t6_class_map.setMaximumHeight(80)
        color_map = {"노란색 (Yellow)": QColor(255, 255, 0, 180), "초록색 (Green)": QColor(0, 255, 0, 180), "빨간색 (Red)": QColor(255, 0, 0, 180), "파란색 (Blue)": QColor(0, 0, 255, 180), "청록색 (Cyan)": QColor(0, 255, 255, 180), "자주색 (Magenta)": QColor(255, 0, 255, 180), "흰색 (White)": QColor(255, 255, 255, 180), "검은색 (Black)": QColor(0, 0, 0, 180)}
        def get_color_icon(color):
            pixmap = QPixmap(16, 16); pixmap.fill(Qt.gray); painter = QPainter(pixmap); painter.fillRect(1, 1, 14, 14, QColor(color.red(), color.green(), color.blue())); painter.end(); return QIcon(pixmap)
        self.t6_color_combo = QComboBox()
        for name, color in color_map.items(): self.t6_color_combo.addItem(get_color_icon(color), name)
        self.t6_color_combo.setCurrentText(ConfigDefaults.TAB6['color'])
        self.t6_color_combo.currentTextChanged.connect(lambda name: self.t6_view.set_crosshair_color(color_map[name]))
        self.t6_btn_load = QPushButton("📂 이미지 목록 불러오기"); self.t6_btn_load.clicked.connect(self.load_labeling_images); self.btn_go_to_preprocess = QPushButton("➡️ 라벨링 완료 (전처리로 이동)"); self.btn_go_to_preprocess.setStyleSheet("background-color: #dbeafe; font-weight: bold; color: #1e3a8a;"); self.btn_go_to_preprocess.clicked.connect(self.go_to_preprocess_tab)
        f.addRow(self.t6_img_dir); f.addRow(self.t6_lbl_dir); f.addRow(QLabel("클래스 목록 (엔터로 구분)"), self.t6_class_map); f.addRow(QLabel("십자선 색상"), self.t6_color_combo)
        split = QSplitter(Qt.Horizontal); left_w = QWidget(); left_l = QVBoxLayout(left_w); left_l.addWidget(self._create_scroll(f))
        
        self.t6_btn_reset = QPushButton("🔄 이 탭 초기화"); self.t6_btn_reset.setStyleSheet("background-color: #fee2e2; border: 1px solid #fca5a5; padding: 5px; border-radius: 4px;"); self.t6_btn_reset.clicked.connect(lambda _, w=left_w: self.reset_tab_defaults(w, "라벨링 툴"))
        h_btns = QHBoxLayout(); h_btns.addWidget(self.t6_btn_load); h_btns.addWidget(self.btn_go_to_preprocess); h_btns.addWidget(self.t6_btn_reset); left_l.addLayout(h_btns)
        
        self.t6_list = QListWidget(); self.t6_list.itemClicked.connect(self.on_label_image_selected); left_l.addWidget(QLabel("<b>이미지 목록</b>")); left_l.addWidget(self.t6_list)
        right_w = QWidget(); right_l = QHBoxLayout(right_w); self.t6_view = LabelingView(); self.t6_view.box_added.connect(self.sync_label_ui); self.t6_view.request_prev.connect(self.select_prev_label_image); self.t6_view.request_next.connect(self.select_next_label_image)
        side_panel = QWidget(); side_layout = QVBoxLayout(side_panel); side_layout.setContentsMargins(10, 0, 0, 0); side_layout.setAlignment(Qt.AlignTop)
        help_text = ("<b>라벨링 영역 단축키</b><br><span style='font-size:11px; color:#555;'><b>Q/E</b>: 실행취소/다시실행<br><b>S/W</b>: OK/NG 클래스 지정<br><b>휠클릭 드래그</b>: 박스 위치이동<br><b>좌클릭 드래그</b>: 크기 조절<br><b>마우스 휠</b>: 화면 확대/축소<br><b>우클릭 드래그</b>: 화면 이동<br><b>Shift/Ctrl+클릭</b>: 다중 선택</span>")
        side_layout.addWidget(QLabel(help_text)); side_layout.addSpacing(15); auto_group = QGroupBox("자동 객체 찾기"); auto_layout = QVBoxLayout(auto_group)
        self.btn_auto_label = QPushButton("✨ 선택 객체 자동 찾기"); self.btn_auto_label.setStyleSheet("background-color: #fef08a; font-weight: bold; color: #854d0e; padding: 5px;"); self.btn_auto_label.clicked.connect(self.run_auto_labeling)
        self.btn_auto_label_next = QPushButton("⏭️ 다음 사진 일괄 적용"); self.btn_auto_label_next.setStyleSheet("background-color: #fed7aa; font-weight: bold; color: #9a3412; padding: 5px;"); self.btn_auto_label_next.clicked.connect(self.run_auto_labeling_next)
        self.t6_auto_thr = QDoubleSpinBox(); self.t6_auto_thr.setRange(0.4, 0.99); self.t6_auto_thr.setValue(ConfigDefaults.TAB6['auto_thr']); self.t6_auto_thr.setSingleStep(0.05); h_thr = QHBoxLayout(); h_thr.addWidget(QLabel("<b>유사도:</b>")); h_thr.addWidget(self.t6_auto_thr); h_thr.addStretch()
        self.t6_auto_nms = QDoubleSpinBox(); self.t6_auto_nms.setRange(0.0, 1.0); self.t6_auto_nms.setValue(ConfigDefaults.TAB6['auto_nms']); self.t6_auto_nms.setSingleStep(0.05); h_nms = QHBoxLayout(); h_nms.addWidget(QLabel("<b>NMS(중복제거):</b>")); h_nms.addWidget(self.t6_auto_nms); h_nms.addStretch()
        self.t6_auto_thr.setProperty("default_val", ConfigDefaults.TAB6['auto_thr'])
        self.t6_auto_nms.setProperty("default_val", ConfigDefaults.TAB6['auto_nms'])
        self.t6_color_combo.setProperty("default_val", ConfigDefaults.TAB6['color'])
        auto_layout.addLayout(h_thr); auto_layout.addLayout(h_nms); auto_layout.addWidget(self.btn_auto_label); auto_layout.addWidget(self.btn_auto_label_next); side_layout.addWidget(auto_group); side_layout.addSpacing(10)
        label_info_group = QGroupBox("현재 라벨 목록"); label_info_layout = QVBoxLayout(label_info_group); label_info_layout.setContentsMargins(5, 5, 5, 5); self.t6_label_list = QListWidget(); self.t6_label_list.setSelectionMode(QListWidget.ExtendedSelection); self.t6_label_list.itemSelectionChanged.connect(self.on_label_selection_changed)
        btn_layout = QHBoxLayout(); self.btn_change_class = QPushButton("🔄 변경"); self.btn_delete_label = QPushButton("🗑️ 삭제"); self.btn_change_class.clicked.connect(self.on_change_class_clicked); self.btn_delete_label.clicked.connect(self.on_delete_label_clicked); btn_layout.addWidget(self.btn_change_class); btn_layout.addWidget(self.btn_delete_label)
        label_info_layout.addWidget(self.t6_label_list, stretch=1); label_info_layout.addLayout(btn_layout); side_layout.addWidget(label_info_group, stretch=1); right_l.addWidget(self.t6_view, stretch=4); right_l.addWidget(side_panel, stretch=1)
        split.addWidget(left_w); split.addWidget(right_w); split.setSizes([300, 1100]); layout = QVBoxLayout(); layout.addWidget(split); tab = QWidget(); tab.setLayout(layout); self.tabs.addTab(tab, "🖌️ 라벨링 툴")

    def go_to_preprocess_tab(self):
        if hasattr(self, 't1_class_map'): self.t1_class_map.setText(self.t6_class_map.toPlainText())
        img_path = Path(self.t6_img_dir.get_path()); parent_dir = img_path.parent 
        if parent_dir.exists():
            self.w_base_ds.line_edit.setText(str(parent_dir)); self.w_proc_ds.line_edit.setText(str(parent_dir / "processed_dataset")); self.tabs.setCurrentIndex(1); self.statusBar().showMessage("➡️ 라벨링 완료!", 5000)

    def browse_tab4_eval_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "평가할 YOLO 모델 선택", str(Path(self.w_work_ds.get_path()) / "runs" / "retrain_train"), "PyTorch Model (*.pt);;All Files (*)")
        if path: self.t4_eval_model_display.setText(path); self.t4_btn_eval.setEnabled(True); self.t4_btn_auto_thr.setEnabled(True); self.statusBar().showMessage(f"📂 모델 로드됨: {Path(path).name}", 3000)

    def export_project_dialog(self):
        logger.info("프로젝트 압축(내보내기) 실행")
        t3_model = self.t3_model.get_path(); t4_model = self.t4_eval_model_display.text(); has_base = t3_model and Path(t3_model).exists(); has_retrained = t4_model and Path(t4_model).exists()
        if not has_base and not has_retrained: QMessageBox.warning(self, "경고", "내보낼 모델이 없습니다."); return
        proj_name = Path(self.w_proj_root.get_path()).name; save_path, _ = QFileDialog.getSaveFileName(self, "프로젝트 내보내기 (Zip)", f"{proj_name}.zip", "Zip Files (*.zip)")
        if not save_path: return
        config_data = self.config_builder.build(self)
        if "global" in config_data and "webhook_url" in config_data["global"]: config_data["global"]["webhook_url"] = ""
        final_json_data = {"metadata": {"saved_at": datetime.now().strftime("%Y%m%d_%H%M%S"), "type": "project_snapshot", "has_base_model": has_base, "has_retrained_model": has_retrained}, "config": config_data}
        self.statusBar().showMessage("📦 프로젝트 내보내기 중...")
        try:
            with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("config.json", json.dumps(final_json_data, ensure_ascii=False, indent=4))
                if has_base: zf.write(t3_model, "base_model.pt")
                if has_retrained: zf.write(t4_model, "retrained_model.pt")
            logger.info(f"프로젝트 내보내기 완료: {save_path}")
            QMessageBox.information(self, "내보내기 완료", f"저장되었습니다:\n{save_path}"); self.statusBar().showMessage(f"✅ 프로젝트 내보내기 완료: {Path(save_path).name}", 5000)
        except Exception as e: 
            logger.error(f"프로젝트 내보내기 중 에러: {e}", exc_info=True)
            QMessageBox.critical(self, "오류", f"오류 발생:\n{e}"); self.statusBar().clearMessage()

    def import_project_dialog(self):
        logger.info("프로젝트 복구(불러오기) 실행")
        load_path, _ = QFileDialog.getOpenFileName(self, "프로젝트 불러오기 (Zip)", "", "Zip Files (*.zip)")
        if not load_path: return
        import_name = Path(load_path).stem; current_root_parent = Path(self.w_proj_root.get_path()).parent; new_proj_path = current_root_parent / import_name; is_renamed = False; original_name = new_proj_path.name; counter = 1
        while new_proj_path.exists(): is_renamed = True; new_proj_path = current_root_parent / f"{import_name}_{counter}"; counter += 1
        new_proj_path.mkdir(parents=True, exist_ok=True); target_dir = new_proj_path / "workspace" / f"Imported_{import_name}"; target_dir.mkdir(parents=True, exist_ok=True)
        self.statusBar().showMessage(f"📥 '{new_proj_path.name}' 프로젝트를 구성하는 중..."); QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            with zipfile.ZipFile(load_path, 'r') as zf:
                resolved_target = Path(target_dir).resolve()
                for member in zf.infolist():
                    member_path = Path(target_dir / member.filename).resolve()
                    if not member_path.is_relative_to(resolved_target):
                        raise PermissionError(f"보안 경고: 압축 파일이 지정된 경로를 벗어나려고 합니다! ({member.filename})")
                zf.extractall(target_dir)

            config_file = target_dir / "config.json"; config_data = {}
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f: config_data = json.load(f).get("config", {})
            self.w_proj_root.line_edit.setText(str(new_proj_path)); self.apply_loaded_config(config_data)
            legacy_model, base_model, retrained_model = target_dir / "model.pt", target_dir / "base_model.pt", target_dir / "retrained_model.pt"
            if base_model.exists() or legacy_model.exists():
                model_path_str = str((base_model if base_model.exists() else legacy_model).resolve()); self.t3_model.line_edit.setText(model_path_str); self.t4_base.line_edit.setText(model_path_str); self.t5_model.line_edit.setText(model_path_str)
            if retrained_model.exists(): retrained_path_str = str(retrained_model.resolve()); self.t4_eval_model_display.setText(retrained_path_str); self.t4_btn_eval.setEnabled(True); self.t4_btn_auto_thr.setEnabled(True); self.t5_model.line_edit.setText(retrained_path_str)
            self.config_manager.update_workspace_path(str(new_proj_path / "workspace"))
            QApplication.restoreOverrideCursor()
            logger.info(f"프로젝트 불러오기 완료: {new_proj_path}")
            if is_renamed: 
                QMessageBox.warning(self, "⚠️ 이름 변경", f"기존에 '{original_name}' 폴더가 존재하여 이름이 변경되었습니다!\n새 폴더명: {new_proj_path.name}"); self.statusBar().showMessage(f"⚠️ 폴더 이름 변경됨: {new_proj_path.name} 로 복구 완료", 7000)
            else: 
                QMessageBox.information(self, "✅ 불러오기 완료", f"루트: {new_proj_path.name}\n모든 경로가 연동되었습니다."); self.statusBar().showMessage(f"✅ 프로젝트 불러오기 완료: {new_proj_path.name}", 5000)
        except Exception as e: 
            logger.error(f"프로젝트 불러오기 중 에러: {e}", exc_info=True)
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "❌ 오류", f"오류 발생:\n{e}"); self.statusBar().clearMessage()
            
    def import_project_dialog(self):
        logger.info("프로젝트 복구(불러오기) 실행")
        load_path, _ = QFileDialog.getOpenFileName(self, "프로젝트 불러오기 (Zip)", "", "Zip Files (*.zip)")
        if not load_path: return
        import_name = Path(load_path).stem; current_root_parent = Path(self.w_proj_root.get_path()).parent; new_proj_path = current_root_parent / import_name; is_renamed = False; original_name = new_proj_path.name; counter = 1
        while new_proj_path.exists(): is_renamed = True; new_proj_path = current_root_parent / f"{import_name}_{counter}"; counter += 1
        new_proj_path.mkdir(parents=True, exist_ok=True); target_dir = new_proj_path / "workspace" / f"Imported_{import_name}"; target_dir.mkdir(parents=True, exist_ok=True)
        self.statusBar().showMessage(f"📥 '{new_proj_path.name}' 프로젝트를 구성하는 중..."); QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            with zipfile.ZipFile(load_path, 'r') as zf:
                resolved_target = Path(target_dir).resolve()
                for member in zf.infolist():
                    member_path = Path(target_dir / member.filename).resolve()
                    if not member_path.is_relative_to(resolved_target):
                        raise PermissionError(f"보안 경고: 압축 파일이 지정된 경로를 벗어나려고 합니다! ({member.filename})")
                zf.extractall(target_dir)

            config_file = target_dir / "config.json"; config_data = {}
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f: config_data = json.load(f).get("config", {})
            self.w_proj_root.line_edit.setText(str(new_proj_path)); self.apply_loaded_config(config_data)
            legacy_model, base_model, retrained_model = target_dir / "model.pt", target_dir / "base_model.pt", target_dir / "retrained_model.pt"
            if base_model.exists() or legacy_model.exists():
                model_path_str = str((base_model if base_model.exists() else legacy_model).resolve()); self.t3_model.line_edit.setText(model_path_str); self.t4_base.line_edit.setText(model_path_str); self.t5_model.line_edit.setText(model_path_str)
            if retrained_model.exists(): retrained_path_str = str(retrained_model.resolve()); self.t4_eval_model_display.setText(retrained_path_str); self.t4_btn_eval.setEnabled(True); self.t4_btn_auto_thr.setEnabled(True); self.t5_model.line_edit.setText(retrained_path_str)
            self.config_manager.update_workspace_path(str(new_proj_path / "workspace"))
            QApplication.restoreOverrideCursor()
            logger.info(f"프로젝트 불러오기 완료: {new_proj_path}")
            if is_renamed: 
                QMessageBox.warning(self, "⚠️ 이름 변경", f"기존에 '{original_name}' 폴더가 존재하여 이름이 변경되었습니다!\n새 폴더명: {new_proj_path.name}"); self.statusBar().showMessage(f"⚠️ 폴더 이름 변경됨: {new_proj_path.name} 로 복구 완료", 7000)
            else: 
                QMessageBox.information(self, "✅ 불러오기 완료", f"루트: {new_proj_path.name}\n모든 경로가 연동되었습니다."); self.statusBar().showMessage(f"✅ 프로젝트 불러오기 완료: {new_proj_path.name}", 5000)
        except Exception as e: 
            logger.error(f"프로젝트 불러오기 중 에러: {e}", exc_info=True)
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "❌ 오류", f"오류 발생:\n{e}"); self.statusBar().clearMessage()

    def setup_tab1(self):
        f = ConfigDefaults.TAB1
        form = QFormLayout(); self.t1_auto_crop = QCheckBox(); self.t1_auto_crop.setChecked(f['auto_crop']); self.t1_margin = QSpinBox(); self.t1_margin.setRange(0, 500); self.t1_margin.setValue(f['margin'])
        h_crop = QHBoxLayout(); self.t1_mx = QSpinBox(); self.t1_my = QSpinBox(); self.t1_mw = QSpinBox(); self.t1_mh = QSpinBox(); self.t1_mw.setRange(1,9999); self.t1_mh.setRange(1,9999); self.t1_mw.setValue(f['mw']); self.t1_mh.setValue(f['mh'])
        for w, l in zip([self.t1_mx, self.t1_my, self.t1_mw, self.t1_mh], ["X", "Y", "W", "H"]): h_crop.addWidget(QLabel(l)); h_crop.addWidget(w)
        self.t1_auto_crop.stateChanged.connect(lambda state: [w.setEnabled(not state) for w in [self.t1_mx, self.t1_my, self.t1_mw, self.t1_mh]])
        [w.setEnabled(False) for w in [self.t1_mx, self.t1_my, self.t1_mw, self.t1_mh]]
        self.t1_mx.setProperty("default_val", 0)
        self.t1_my.setProperty("default_val", 0)
        self.t1_mw.setProperty("default_val", f['mw'])
        self.t1_mh.setProperty("default_val", f['mh'])
        self.t1_class_map = QTextEdit(); self.t1_class_map.setPlainText("OK\nNG"); self.t1_class_map.setMaximumHeight(80); self.t1_clean = QCheckBox(); self.t1_clean.setChecked(f['clean']); self.t1_exif = QCheckBox(); self.t1_exif.setChecked(f['exif'])
        self.add_param(form, "자동 크롭", self.t1_auto_crop, f['auto_crop']); self.add_param(form, "자동 크롭 여백(px)", self.t1_margin, f['margin']); form.addRow("수동 크롭 영역", h_crop)
        self.btn_scan_classes = QPushButton("🔍 데이터셋에서 클래스 자동 추출"); self.btn_scan_classes.setStyleSheet("background-color: #dbeafe; font-weight: bold; color: #1e3a8a; padding: 5px;"); self.btn_scan_classes.clicked.connect(self.scan_classes_from_data)
        v_class_layout = QVBoxLayout(); v_class_layout.addWidget(self.t1_class_map); v_class_layout.addWidget(self.btn_scan_classes); form.addRow(QLabel("클래스 목록\n(엔터로 구분)"), v_class_layout)
        self.add_param(form, "기존 출력 폴더 초기화", self.t1_clean, f['clean']); self.add_param(form, "EXIF 회전 보정", self.t1_exif, f['exif'])
        self.t1_btn_run = QPushButton("🚀 전처리 시작"); self.t1_btn_run.clicked.connect(self.run_tab1); self.t1_progress = QProgressBar(); self.t1_log = QTextEdit(); self.t1_log.setReadOnly(True); self.t1_img_grid = ImageGridWidget(max_display=100)
        split = QSplitter(Qt.Horizontal); left_w = QWidget(); left_l = QVBoxLayout(left_w); left_l.addWidget(self._create_scroll(form))
        
        h_t1_btns = QHBoxLayout(); h_t1_btns.addWidget(self.t1_btn_run)
        self.t1_btn_reset = QPushButton("🔄 이 탭 초기화"); self.t1_btn_reset.setStyleSheet("background-color: #fee2e2; border: 1px solid #fca5a5; padding: 5px; border-radius: 4px;")
        self.t1_btn_reset.clicked.connect(lambda _, w=left_w: self.reset_tab_defaults(w, "데이터 전처리")); h_t1_btns.addWidget(self.t1_btn_reset)
        left_l.addLayout(h_t1_btns); left_l.addWidget(self.t1_progress); left_l.addWidget(self.t1_log)
        
        right_w = QWidget(); right_l = QVBoxLayout(right_w); right_l.addWidget(QLabel("<b>전처리 결과 이미지 미리보기</b>")); right_l.addWidget(self.t1_img_grid)
        split.addWidget(left_w); split.addWidget(right_w); split.setSizes([600, 800]); layout = QVBoxLayout(); layout.addWidget(split); tab = QWidget(); tab.setLayout(layout); self.tabs.addTab(tab, "✂️ 데이터 전처리")

    def run_tab1(self):
        logger.info("Tab1 데이터 전처리 버튼 클릭됨")
        base_dir = Path(self.w_base_ds.get_path())
        is_valid, err = self.validate_paths(원본_데이터_dir=base_dir, 이미지_dir=base_dir/"image", 라벨_dir=base_dir/"data")
        if not is_valid: QMessageBox.warning(self, "경로 오류", err); return
        success, cmap_or_error = self.parse_and_validate_class_map(self.t1_class_map.toPlainText())
        if not success: QMessageBox.warning(self, "클래스 매핑 오류", cmap_or_error); return
        config = {"base_dir": self.w_base_ds.get_path(), "processed_dir": self.w_proc_ds.get_path(), "use_auto_crop": self.t1_auto_crop.isChecked(), "margin": self.t1_margin.value(), "manual_crop": (self.t1_mx.value(), self.t1_my.value(), self.t1_mw.value(), self.t1_mh.value()), "class_map": cmap_or_error, "clean_old": self.t1_clean.isChecked(), "use_exif": self.t1_exif.isChecked(), "webhook_url": self.webhook_url, "noti_flags": self.get_noti_flags()}
        
        if self.webhook_url and self.noti_flags.get("start"):
            send_discord_webhook(
                webhook_url=self.webhook_url,
                title="▶️ [작업 시작]",
                description="데이터 전처리 작업이 시작되었습니다.",
                color=0x9b59b6
            )
        
        self.t1_btn_run.setEnabled(False); self.t1_log.clear(); self.statusBar().showMessage("✂️ 데이터 전처리 진행 중... 잠시만 기다려주세요."); QApplication.setOverrideCursor(Qt.WaitCursor)
        self.t1_thread = PreprocessThread(config); self.t1_thread.progress.connect(self.t1_progress.setValue); self.t1_thread.log_msg.connect(self.t1_log.append); self.t1_thread.error.connect(lambda e: self.on_thread_error("데이터 전처리", e)); self.t1_thread.finished_ok.connect(self.on_tab1_finished)
        self.t1_thread.finished.connect(lambda: [self.t1_btn_run.setEnabled(True), QApplication.restoreOverrideCursor(), self.statusBar().showMessage("✅ 전처리 완료", 5000)]); self.t1_thread.start()

    def on_tab1_finished(self, ok_count):
        if self.webhook_url and self.noti_flags.get("task"):
            send_discord_webhook(
                webhook_url=self.webhook_url,
                title="✅ [작업 완료] 데이터 전처리",
                description=f"총 **{ok_count}장**이 성공적으로 처리되었습니다.",
                color=0x2ecc71
            )
        QMessageBox.information(self, "완료", f"{ok_count}장 전처리 완료!")
        proc_preview_dir = Path(self.w_proc_ds.get_path()) / "preview"
        target_dir = proc_preview_dir if proc_preview_dir.exists() else Path(self.w_proc_ds.get_path()) / "images"
        if target_dir.exists(): self.t1_img_grid.update_images([str(f) for f in target_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}])

    def setup_tab2(self):
        d = ConfigDefaults.TAB2
        self.t2_epochs = QSpinBox(); self.t2_epochs.setRange(1, 5000); self.t2_epochs.setValue(d['epochs'])
        self.t2_batch = QSpinBox(); self.t2_batch.setRange(1, 256); self.t2_batch.setValue(d['batch'])
        self.t2_workers = QSpinBox(); self.t2_workers.setRange(0, 32); self.t2_workers.setValue(d['workers'])
        self.t2_patience = QSpinBox(); self.t2_patience.setRange(0, 10000); self.t2_patience.setValue(d['patience'])
        self.t2_seed = QSpinBox(); self.t2_seed.setRange(0, 999999); self.t2_seed.setValue(d['seed'])
        self.t2_folds = QSpinBox(); self.t2_folds.setRange(1, 10); self.t2_folds.setValue(d['folds'])
        self.t2_test_split = QDoubleSpinBox(); self.t2_test_split.setRange(0.05, 0.6); self.t2_test_split.setValue(d['test_split']); self.t2_test_split.setSingleStep(0.05)
        
        self.t2_lcls = QDoubleSpinBox(); self.t2_lcls.setRange(0.1, 10.0); self.t2_lcls.setValue(d['lcls']); self.t2_lcls.setSingleStep(0.1)
        self.t2_lbox = QDoubleSpinBox(); self.t2_lbox.setRange(0.1, 20.0); self.t2_lbox.setValue(d['lbox']); self.t2_lbox.setSingleStep(0.5)
        self.t2_ldfl = QDoubleSpinBox(); self.t2_ldfl.setRange(0.1, 10.0); self.t2_ldfl.setValue(d['ldfl']); self.t2_ldfl.setSingleStep(0.1)
        
        def make_dbl(rng, val, step): b = QDoubleSpinBox(); b.setRange(*rng); b.setDecimals(3 if step < 0.01 else 2); b.setSingleStep(step); b.setValue(val); return b
        
        self.t2_ah = make_dbl((0, 0.1), d['ah'], 0.001); self.t2_as = make_dbl((0, 1.0), d['as'], 0.05); self.t2_av = make_dbl((0, 1.0), d['av'], 0.05)
        self.t2_adeg = make_dbl((0, 45.0), d['adeg'], 1.0); self.t2_atrans = make_dbl((0, 0.5), d['atrans'], 0.01); self.t2_ascale = make_dbl((0, 1.0), d['ascale'], 0.05)
        self.t2_ashear = make_dbl((0, 30.0), d['ashear'], 1.0); self.t2_afud = make_dbl((0, 1.0), d['afud'], 0.05); self.t2_aflr = make_dbl((0, 1.0), d['aflr'], 0.05)
        self.t2_amos = make_dbl((0, 1.0), d['amos'], 0.05); self.t2_amix = make_dbl((0, 1.0), d['amix'], 0.05); self.t2_acp = make_dbl((0, 1.0), d['acp'], 0.05)
        
        self.t2_tune_iterations = QSpinBox(); self.t2_tune_iterations.setRange(10, 300); self.t2_tune_iterations.setValue(d['tune_iterations'])
        
        h_form = QHBoxLayout(); f_left = QFormLayout(); f_right = QFormLayout()

        f_left.addRow(QLabel("<b>[기본 파라미터]</b>"))
        self.add_param(f_left, "Epochs", self.t2_epochs, d['epochs']); self.add_param(f_left, "Batch", self.t2_batch, d['batch']); self.add_param(f_left, "Workers", self.t2_workers, d['workers']); self.add_param(f_left, "Patience (조기 종료)", self.t2_patience, d['patience']); self.add_param(f_left, "Random Seed", self.t2_seed, d['seed']); self.add_param(f_left, "Fold 수", self.t2_folds, d['folds']); self.add_param(f_left, "Test 분리 비율", self.t2_test_split, d['test_split'])
        f_left.addRow(QLabel("<br><b>[Loss 가중치]</b>"))
        self.add_param(f_left, "cls", self.t2_lcls, d['lcls']); self.add_param(f_left, "box", self.t2_lbox, d['lbox']); self.add_param(f_left, "dfl", self.t2_ldfl, d['ldfl'])
        f_left.addRow(QLabel("<br><b>[Auto ML 최적화]</b>"))
        self.add_param(f_left, "튜닝 반복 횟수 (Iterations)", self.t2_tune_iterations, d['tune_iterations'])

        f_right.addRow(QLabel("<b>[데이터 증강]</b>"))
        self.add_param(f_right, "HSV(H)", self.t2_ah, d['ah']); self.add_param(f_right, "HSV(S)", self.t2_as, d['as']); self.add_param(f_right, "HSV(V)", self.t2_av, d['av']); self.add_param(f_right, "Degrees", self.t2_adeg, d['adeg']); self.add_param(f_right, "Translate", self.t2_atrans, d['atrans']); self.add_param(f_right, "Scale", self.t2_ascale, d['ascale']); self.add_param(f_right, "Shear", self.t2_ashear, d['ashear']); self.add_param(f_right, "Flip UD", self.t2_afud, d['afud']); self.add_param(f_right, "Flip LR", self.t2_aflr, d['aflr']); self.add_param(f_right, "Mosaic", self.t2_amos, d['amos']); self.add_param(f_right, "Mixup", self.t2_amix, d['amix']); self.add_param(f_right, "Copy-Paste", self.t2_acp, d['acp'])
        
        h_form.addLayout(f_left); h_form.addLayout(f_right)
        
        h_tune = QHBoxLayout()
        self.t2_btn_tune = QPushButton("🤖 최적 파라미터 자동 탐색 (Auto ML)")
        self.t2_btn_tune.setStyleSheet("background-color: #dbeafe; border: 1px solid #93c5fd; padding: 8px; font-weight: bold; color: #1e3a8a;")
        self.t2_btn_tune.clicked.connect(self.run_auto_tune)
        
        self.btn_show_tune_history = QPushButton("📈 AutoML 튜닝 기록 그래프 보기")
        self.btn_show_tune_history.setStyleSheet("background-color: #fce7f3; border: 1px solid #fbcfe8; padding: 8px; font-weight: bold; color: #9d174d;")
        self.btn_show_tune_history.clicked.connect(self.show_tune_history)
        
        h_tune.addWidget(self.t2_btn_tune)
        h_tune.addWidget(self.btn_show_tune_history)
        
        self.t2_btn_run = QPushButton("🚀 K-Fold 학습 시작"); self.t2_btn_run.clicked.connect(self.run_tab2)
        
        self.t2_btn_stop = QPushButton("🛑 안전 종료(Graceful Stop)"); self.t2_btn_stop.clicked.connect(self.stop_training); self.t2_btn_stop.setEnabled(False)
        
        l = QVBoxLayout(); self.t2_scroll = self._create_scroll(h_form); l.addWidget(self.t2_scroll)
        
        self.t2_btn_reset = QPushButton("🔄 이 탭 초기화"); self.t2_btn_reset.setStyleSheet("background-color: #fee2e2; border: 1px solid #fca5a5; padding: 5px; border-radius: 4px;"); self.t2_btn_reset.clicked.connect(lambda _, w=self.t2_scroll: self.reset_tab_defaults(w, "K-Fold 학습"))
        
        l.addLayout(h_tune)
        
        h = QHBoxLayout(); h.addWidget(self.t2_btn_run); h.addWidget(self.t2_btn_stop); h.addWidget(self.t2_btn_reset); l.addLayout(h)
        
        tab = QWidget(); tab.setLayout(l); self.tabs.addTab(tab, "🏋️ K-Fold 학습")

    def show_tune_history(self):
        history_path = Path(self.w_work_ds.get_path()) / "runs" / "tune_custom" / "tune_history.csv"
        if not history_path.exists():
            QMessageBox.warning(self, "기록 없음", "저장된 튜닝 기록(tune_history.csv)이 없습니다.\n먼저 Auto ML 탐색을 실행해주세요.")
            return
        dialog = TuneHistoryDialog(history_path, self)
        dialog.exec_()

    def run_auto_tune(self):
        logger.info("Tab2 Auto ML 파라미터 튜닝 실행")
        if self.training_process and self.training_process.is_alive(): QMessageBox.warning(self, "경고", "이미 진행 중인 프로세스가 있습니다."); return
        proc_dir = Path(self.w_proc_ds.get_path()); is_valid, err = self.validate_paths(처리된_데이터_dir=proc_dir, 이미지_dir=proc_dir/"images", 라벨_dir=proc_dir/"labels")
        if not is_valid: QMessageBox.warning(self, "경로 오류", err); return
        success, cmap_or_error = self.parse_and_validate_class_map(self.t1_class_map.toPlainText())
        if not success: QMessageBox.warning(self, "클래스 매핑 오류", cmap_or_error); return
        if QMessageBox.question(self, 'Auto ML 튜닝', f'총 {self.t2_tune_iterations.value()}번의 조합 탐색을 시작합니다.\n계속 진행하시겠습니까?', QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
        
        initial_params = {'box': self.t2_lbox.value(), 'cls': self.t2_lcls.value(), 'dfl': self.t2_ldfl.value(), 'hsv_h': self.t2_ah.value(), 'hsv_s': self.t2_as.value(), 'hsv_v': self.t2_av.value(), 'degrees': self.t2_adeg.value(), 'translate': self.t2_atrans.value(), 'scale': self.t2_ascale.value(), 'shear': self.t2_ashear.value(), 'flipud': self.t2_afud.value(), 'fliplr': self.t2_aflr.value(), 'mosaic': self.t2_amos.value(), 'mixup': self.t2_amix.value(), 'copy_paste': self.t2_acp.value()}
        
        args = {
            "processed_dir": self.w_proc_ds.get_path(), "workspace_dir": self.w_work_ds.get_path(), 
            "model_name": self.g_model.currentText(), "epochs": self.t2_epochs.value(), 
            "iterations": self.t2_tune_iterations.value(), "batch": self.t2_batch.value(), 
            "workers": self.t2_workers.value(), "class_names": list(cmap_or_error.keys()), 
            "initial_params": initial_params, "match_iou": getattr(self, 't3_match_iou', QDoubleSpinBox()).value(), 
            "webhook_url": self.webhook_url, "noti_flags": self.get_noti_flags(),
            "tune_epochs": 30, "tune_patience": 5
        }
        self.start_training_process(_tune_worker, args)

    def run_tab2(self):
        logger.info("Tab2 K-Fold 학습 실행 버튼 클릭")
        if self.training_process and self.training_process.is_alive(): QMessageBox.warning(self, "경고", "이미 진행 중인 프로세스가 있습니다."); return
        proc_dir = Path(self.w_proc_ds.get_path()); is_valid, err = self.validate_paths(처리된_데이터_dir=proc_dir, 이미지_dir=proc_dir/"images", 라벨_dir=proc_dir/"labels")
        if not is_valid: QMessageBox.warning(self, "경로 오류", err); return
        if not self.g_model.currentText(): QMessageBox.warning(self, "입력 오류", "모델을 선택해주세요."); return
        if self.t2_folds.value() == 1:
            if QMessageBox.question(self, '확인', 'Fold 수가 1입니다. 단일 분할 학습으로 진행하시겠습니까?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.No: return
        success, cmap_or_error = self.parse_and_validate_class_map(self.t1_class_map.toPlainText())
        if not success: QMessageBox.warning(self, "클래스 매핑 오류", cmap_or_error); return
        args = {"processed_dir": self.w_proc_ds.get_path(), "workspace_dir": self.w_work_ds.get_path(), "webhook_url": self.webhook_url, "noti_flags": self.get_noti_flags(), "model_name": self.g_model.currentText(), "imgsz": int(self.g_imgsz.currentText()), "epochs": self.t2_epochs.value(), "batch": self.t2_batch.value(), "workers": self.t2_workers.value(), "patience": self.t2_patience.value(), "random_seed": self.t2_seed.value(), "deterministic": False, "num_folds": self.t2_folds.value(), "test_split": self.t2_test_split.value(), "best_metric": "metrics/mAP50-95(B)", "second_metric": "metrics/mAP50(B)", "class_names": list(cmap_or_error.keys()), "aug": {"hsv_h": self.t2_ah.value(), "hsv_s": self.t2_as.value(), "hsv_v": self.t2_av.value(), "degrees": self.t2_adeg.value(), "translate": self.t2_atrans.value(), "scale": self.t2_ascale.value(), "shear": self.t2_ashear.value(), "flipud": self.t2_afud.value(), "fliplr": self.t2_aflr.value(), "mosaic": self.t2_amos.value(), "mixup": self.t2_amix.value(), "copy_paste": self.t2_acp.value()}, "loss": {"cls": self.t2_lcls.value(), "box": self.t2_lbox.value(), "dfl": self.t2_ldfl.value()}, "match_iou": getattr(self, 't3_match_iou', QDoubleSpinBox()).value()}
        self.start_training_process(_kfold_train_worker, args)

    def setup_tab3(self):
        d = ConfigDefaults.TAB3
        f = QFormLayout(); proc_dir, work_dir = Path(self.w_proc_ds.get_path()), Path(self.w_work_ds.get_path()); self.t3_model = PathInputWidget("평가 모델 (.pt)", False, str(work_dir/"kfold"/"best_model.pt")); f.addRow(self.t3_model); self.t3_img = PathInputWidget("평가 이미지", True, str(proc_dir/"images")); f.addRow(self.t3_img); self.t3_lbl = PathInputWidget("정답 라벨", True, str(proc_dir/"labels")); f.addRow(self.t3_lbl)
        
        self.t3_conf = QDoubleSpinBox(); self.t3_conf.setRange(0.01, 0.99); self.t3_conf.setValue(d['conf']); self.t3_conf.setSingleStep(0.01)
        self.t3_iou = QDoubleSpinBox(); self.t3_iou.setRange(0.01, 0.99); self.t3_iou.setValue(d['iou']); self.t3_iou.setSingleStep(0.01)
        self.t3_match_iou = QDoubleSpinBox(); self.t3_match_iou.setRange(0.01, 0.99); self.t3_match_iou.setValue(d['match_iou']); self.t3_match_iou.setSingleStep(0.01)
        self.t3_max_det = QSpinBox(); self.t3_max_det.setRange(1, 999); self.t3_max_det.setValue(d['max_det'])
        self.t3_run_name = QLineEdit(d['run_name'])
        self.t3_agnostic = QCheckBox(); self.t3_agnostic.setChecked(d['agnostic'])
        self.t3_save_rel = QCheckBox(); self.t3_save_rel.setChecked(d['save_rel'])
        
        self.add_param(f, "Confidence Threshold", self.t3_conf, d['conf']); self.add_param(f, "IoU (NMS)", self.t3_iou, d['iou']); self.add_param(f, "정답 매칭 IoU", self.t3_match_iou, d['match_iou']); self.add_param(f, "최대 탐지 수", self.t3_max_det, d['max_det']); self.add_param(f, "실행 이름", self.t3_run_name, d['run_name']); self.add_param(f, "Agnostic NMS", self.t3_agnostic, d['agnostic']); self.add_param(f, "오답 별도 저장 (재학습용)", self.t3_save_rel, d['save_rel'])
        
        self.t3_btn_run = QPushButton("🔍 평가 및 오답 선별 실행"); self.t3_btn_run.clicked.connect(self.run_tab3)
        self.t3_btn_auto_thr = QPushButton("🎯 스레숄드 찾기"); self.t3_btn_auto_thr.setStyleSheet("background-color: #fef08a; font-weight: bold; color: #854d0e;"); self.t3_btn_auto_thr.clicked.connect(self.run_auto_threshold)
        self.btn_send_t3_to_t5 = QPushButton("➡️ 이 모델과 설정으로 거리 측정"); self.btn_send_t3_to_t5.setStyleSheet("background-color: #dbeafe; font-weight: bold; color: #1e3a8a;"); self.btn_send_t3_to_t5.clicked.connect(lambda: self.send_to_measure_tab(self.t3_model.get_path(), self.t3_img.get_path(), self.t3_conf.value(), self.t3_iou.value(), self.t3_max_det.value(), self.t3_agnostic.isChecked()))
        
        self.t3_table = QTableWidget(0, 5); self.t3_table.setHorizontalHeaderLabels(["파일명", "상태", "예측 수", "정답 수", "사유"]); self._apply_table_style(self.t3_table); header = self.t3_table.horizontalHeader()
        for i in range(self.t3_table.columnCount()): header.setSectionResizeMode(i, QHeaderView.Stretch)
        self.t3_table.setSortingEnabled(True); self.t3_table.itemDoubleClicked.connect(self.on_t3_table_double_clicked)
        
        self.t3_img_grid = ImageGridWidget(max_display=100); self.t3_chk_show_all = QCheckBox("전체 평가 이미지 보기"); self.t3_chk_show_all.setEnabled(False); self.t3_chk_show_all.stateChanged.connect(self.update_tab3_visualization)
        
        split = QSplitter(Qt.Horizontal); t_w = QWidget(); t_l = QVBoxLayout(t_w); t_l.addWidget(self._create_scroll(f))
        
        self.t3_btn_reset = QPushButton("🔄 이 탭 초기화"); self.t3_btn_reset.setStyleSheet("background-color: #fee2e2; border: 1px solid #fca5a5; padding: 5px; border-radius: 4px;"); self.t3_btn_reset.clicked.connect(lambda _, w=t_w: self.reset_tab_defaults(w, "평가 & 선별"))
        
        h_t3_btns = QHBoxLayout(); h_t3_btns.addWidget(self.t3_btn_run); h_t3_btns.addWidget(self.t3_btn_auto_thr); h_t3_btns.addWidget(self.t3_btn_reset)
        t_l.addLayout(h_t3_btns); t_l.addWidget(self.btn_send_t3_to_t5); t_l.addWidget(QLabel("<b>평가 결과 목록</b>")); t_l.addWidget(self.t3_table)
        
        i_w = QWidget(); i_l = QVBoxLayout(i_w); i_header = QHBoxLayout(); i_header.addWidget(QLabel("<b>예측 결과 시각화</b>")); i_header.addStretch(1); i_header.addWidget(self.t3_chk_show_all); i_l.addLayout(i_header); i_l.addWidget(self.t3_img_grid)
        split.addWidget(t_w); split.addWidget(i_w); split.setSizes([700, 700]); l = QVBoxLayout(); l.addWidget(split); tab = QWidget(); tab.setLayout(l); self.tabs.addTab(tab, "🔍 평가 & 선별")

    def run_auto_threshold(self):
        logger.info("Tab3 Auto Threshold 실행 버튼 클릭")
        if self.training_process and self.training_process.is_alive(): QMessageBox.warning(self, "경고", "이미 진행 중입니다."); return
        is_valid, err = self.validate_paths(평가모델_file=Path(self.t3_model.get_path()), 평가이미지_dir=Path(self.t3_img.get_path()), 정답라벨_dir=Path(self.t3_lbl.get_path()))
        if not is_valid: QMessageBox.warning(self, "경로 오류", err); return
        
        msg = ('설정된 [정답 매칭 IoU] 기준에 맞춰, 모델의 성능을 최대로 만드는\n'
               '최적의 Confidence와 NMS IoU 값을 자동 탐색합니다.\n\n'
               '💡 탐색 로직 안내:\n'
               '- 1순위: 완벽 정답 이미지 수 최대화\n'
               '- 2순위: F1-Score 최대화\n'
               '- 1차 성근 탐색 후 최적점 근처 2차 정밀 탐색을 진행합니다.\n\n'
               '진행하시겠습니까?')
        if QMessageBox.question(self, '스레숄드 자동 탐색', msg, QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
        
        args = {"model_path": self.t3_model.get_path(), "img_dir": self.t3_img.get_path(), "lbl_dir": self.t3_lbl.get_path(), "match_iou": self.t3_match_iou.value(), "agnostic": self.t3_agnostic.isChecked(), "max_det": self.t3_max_det.value()}
        self.start_training_process(_auto_threshold_worker, args)

    def run_tab3(self):
        logger.info("Tab3 평가 및 오답 선별 실행 버튼 클릭")
        is_valid, err = self.validate_paths(평가모델_file=Path(self.t3_model.get_path()), 평가이미지_dir=Path(self.t3_img.get_path()), 정답라벨_dir=Path(self.t3_lbl.get_path()))
        if not is_valid: QMessageBox.warning(self, "경로 오류", err); return
        config = {"eval_model_path": self.t3_model.get_path(), "eval_source": self.t3_img.get_path(), "gt_labels_path": self.t3_lbl.get_path(), "workspace_dir": self.w_work_ds.get_path(), "eval_run_name": self.t3_run_name.text(), "eval_conf": self.t3_conf.value(), "eval_iou": self.t3_iou.value(), "match_iou": self.t3_match_iou.value(), "max_det": self.t3_max_det.value(), "agnostic_nms": self.t3_agnostic.isChecked(), "save_relabel": self.t3_save_rel.isChecked(), "webhook_url": self.webhook_url, "noti_flags": self.get_noti_flags()}
        
        if self.webhook_url and self.noti_flags.get("start"): 
            send_discord_webhook(
                webhook_url=self.webhook_url,
                title="▶️ [작업 시작]",
                description="모델 평가 및 오답 선별 작업이 시작되었습니다.",
                color=0x9b59b6
            )
        
        self.t3_btn_run.setEnabled(False); self.statusBar().showMessage("🔍 평가 진행 중..."); QApplication.setOverrideCursor(Qt.WaitCursor)
        self.t3_thread = EvalThread(config); self.t3_thread.finished_ok.connect(self.on_tab3_finished); self.t3_thread.error.connect(lambda e: self.on_thread_error("모델 평가", e)); self.t3_thread.finished.connect(lambda: [self.t3_btn_run.setEnabled(True), QApplication.restoreOverrideCursor(), self.statusBar().clearMessage()]); self.t3_thread.start()

    def on_tab3_finished(self, df, wrong_imgs, all_imgs, stats):
        self.t3_last_wrong_imgs = wrong_imgs; self.t3_last_all_imgs = all_imgs; self.t3_table.setSortingEnabled(False); self.t3_table.setRowCount(len(df))     
        for i, row in df.iterrows():
            self.t3_table.setItem(i, 0, QTableWidgetItem(str(row["파일명"]))); self.t3_table.setItem(i, 1, QTableWidgetItem(str(row["상태"]))); self.t3_table.setItem(i, 2, QTableWidgetItem(str(row["예측 수"]))); self.t3_table.setItem(i, 3, QTableWidgetItem(str(row["정답 수"]))); self.t3_table.setItem(i, 4, QTableWidgetItem(str(row["사유"])))
            for col in range(5): self.t3_table.item(i, col).setTextAlignment(Qt.AlignCenter)
        self.t3_table.setSortingEnabled(True); self.t3_chk_show_all.setEnabled(True); self.t3_chk_show_all.blockSignals(True); self.t3_chk_show_all.setChecked(False); self.t3_chk_show_all.blockSignals(False); self.update_tab3_visualization()
        
        eval_path = self.t3_model.get_path(); display_name = Path(eval_path).name
        if display_name == "best_model.pt":
            source_txt = Path(eval_path).parent / "best_model_source.txt"
            if source_txt.exists(): display_name += f" (원본: {source_txt.read_text(encoding='utf-8').strip()})"
        model_name_with_path = f"{display_name}  |  {eval_path}"
        self.log_db.insert_eval_log(task_type="Tab 3 Eval", model_name=model_name_with_path, total=stats['total'], wrong=stats['wrong'], accuracy=stats['acc'], wrong_imgs_list=wrong_imgs, config_data=self.config_builder.build(self))
        
        if self.webhook_url and self.noti_flags.get("task"):
            send_discord_webhook(
                webhook_url=self.webhook_url,
                title="✅ [작업 완료] 모델 평가",
                description=f"전체 {stats['total']}장 중 {stats['wrong']}장 오답 (정확도 {stats['acc']:.1f}%)",
                color=0x2ecc71
            )
        
        QMessageBox.information(self, "평가 완료", f"총 {stats['total']}장 중 {stats['correct']}장 완벽 일치, {stats['wrong']}장 이상 발생\n\n[객체 단위 검출 성능]\n- 정밀도(Precision): {stats.get('precision', 0):.1f}%\n- 재현율(Recall): {stats.get('recall', 0):.1f}%\n- F1-Score: {stats.get('f1_score', stats['acc']):.1f}%")
        if stats["wrong"] > 0 and stats["relabel_dir"]: self.t4_hard.line_edit.setText(stats["relabel_dir"]); self.t4_orig.line_edit.setText(self.t3_lbl.get_path()); self.t4_base.line_edit.setText(self.t3_model.get_path())

    def update_tab3_visualization(self):
        if not hasattr(self, 't3_last_all_imgs'): return
        self.t3_img_grid.update_images(self.t3_last_all_imgs if self.t3_chk_show_all.isChecked() else self.t3_last_wrong_imgs)

    def on_t3_table_double_clicked(self, item):
        if not hasattr(self, 't3_last_all_imgs'): return
        file_name = self.t3_table.item(item.row(), 0).text()
        try: index = next(i for i, path in enumerate(self.t3_last_all_imgs) if Path(path).name == file_name); ImagePreviewDialog(self.t3_last_all_imgs, index, self).exec_()
        except StopIteration: QMessageBox.warning(self, "오류", "파일을 찾을 수 없습니다.")

    def setup_tab4(self):
        d = ConfigDefaults.TAB4
        main_split = QSplitter(Qt.Horizontal); left_widget = QWidget(); left_layout = QVBoxLayout(left_widget); left_layout.setContentsMargins(0, 0, 0, 0); sub_tabs = QTabWidget(); left_layout.addWidget(sub_tabs)
        sub_tab1 = QWidget(); st1_layout = QVBoxLayout(sub_tab1); f1 = QFormLayout(); proc_dir, work_dir = Path(self.w_proc_ds.get_path()), Path(self.w_work_ds.get_path())
        
        self.t4_hard = PathInputWidget("오답(Hard) 폴더", True, ""); self.t4_orig = PathInputWidget("원본 정답 폴더", True, str(proc_dir/"labels")); self.t4_base = PathInputWidget("베이스 모델 (.pt)", False, str(work_dir/"kfold"/"best_model.pt"))
        f1.addRow(self.t4_hard); f1.addRow(self.t4_orig); f1.addRow(self.t4_base)
        
        self.t4_epochs = QSpinBox(); self.t4_epochs.setRange(1, 1000); self.t4_epochs.setValue(d['epochs'])
        self.t4_batch = QSpinBox(); self.t4_batch.setRange(1, 256); self.t4_batch.setValue(d['batch'])
        self.t4_run = QLineEdit(d['run'])
        self.t4_lcls = QDoubleSpinBox(); self.t4_lcls.setRange(0.1, 10.0); self.t4_lcls.setValue(d['lcls']); self.t4_lcls.setSingleStep(0.1)
        self.t4_lbox = QDoubleSpinBox(); self.t4_lbox.setRange(0.1, 20.0); self.t4_lbox.setValue(d['lbox']); self.t4_lbox.setSingleStep(0.5)
        
        def make_dbl(rng, val, step): b = QDoubleSpinBox(); b.setRange(*rng); b.setDecimals(3 if step < 0.01 else 2); b.setSingleStep(step); b.setValue(val); return b
        
        self.t4_ah = make_dbl((0, 0.1), d['ah'], 0.001); self.t4_as = make_dbl((0, 1.0), d['as'], 0.05); self.t4_av = make_dbl((0, 1.0), d['av'], 0.05)
        self.t4_afud = make_dbl((0, 1.0), d['afud'], 0.05); self.t4_aflr = make_dbl((0, 1.0), d['aflr'], 0.05); self.t4_amos = make_dbl((0, 1.0), d['amos'], 0.05)
        self.t4_amix = make_dbl((0, 1.0), d['amix'], 0.05); self.t4_acp = make_dbl((0, 1.0), d['acp'], 0.05)
        
        f1.addRow(QLabel("<b>[기본 설정]</b>")); self.add_param(f1, "Epochs", self.t4_epochs, d['epochs']); self.add_param(f1, "Batch", self.t4_batch, d['batch']); self.add_param(f1, "Run Name", self.t4_run, d['run']); self.add_param(f1, "cls Loss", self.t4_lcls, d['lcls']); self.add_param(f1, "box Loss", self.t4_lbox, d['lbox'])
        f1.addRow(QLabel("<br><b>[재학습 증강 설정]</b>")); self.add_param(f1, "HSV(H)", self.t4_ah, d['ah']); self.add_param(f1, "HSV(S)", self.t4_as, d['as']); self.add_param(f1, "HSV(V)", self.t4_av, d['av']); self.add_param(f1, "Flip UD", self.t4_afud, d['afud']); self.add_param(f1, "Flip LR", self.t4_aflr, d['aflr']); self.add_param(f1, "Mosaic", self.t4_amos, d['amos']); self.add_param(f1, "Mixup", self.t4_amix, d['amix']); self.add_param(f1, "Copy-Paste", self.t4_acp, d['acp'])
        
        self.t4_scroll = self._create_scroll(f1); st1_layout.addWidget(self.t4_scroll); self.t4_btn_retrain = QPushButton("🔁 Hard Example 재학습 시작"); self.t4_btn_retrain.setMinimumHeight(40); self.t4_btn_retrain.clicked.connect(self.run_tab4_retrain); st1_layout.addWidget(self.t4_btn_retrain)
        
        sub_tab2 = QWidget(); st2_layout = QVBoxLayout(sub_tab2); eval_group = QGroupBox("최종 평가 설정"); eval_layout = QVBoxLayout(eval_group); f2 = QFormLayout()
        
        self.t4_conf = QDoubleSpinBox(); self.t4_conf.setRange(0.01, 0.99); self.t4_conf.setValue(d['eval_conf']); self.t4_conf.setSingleStep(0.01)
        self.t4_iou = QDoubleSpinBox(); self.t4_iou.setRange(0.01, 0.99); self.t4_iou.setValue(d['eval_iou']); self.t4_iou.setSingleStep(0.01)
        self.t4_match_iou = QDoubleSpinBox(); self.t4_match_iou.setRange(0.01, 0.99); self.t4_match_iou.setValue(d['eval_match']); self.t4_match_iou.setSingleStep(0.01)
        self.t4_max_det = QSpinBox(); self.t4_max_det.setRange(1, 999); self.t4_max_det.setValue(d['eval_max'])
        self.t4_agnostic = QCheckBox(); self.t4_agnostic.setChecked(d['eval_agnostic'])
        
        h_eval_model = QHBoxLayout(); self.t4_eval_model_display = QLineEdit(); self.t4_eval_model_display.setReadOnly(True); self.t4_eval_model_display.setPlaceholderText("모델을 선택하거나 첫 번째 탭에서 재학습을 완료하세요."); self.t4_eval_model_display.setStyleSheet("background-color: #f3f4f6; color: #374151; font-weight: bold;"); self.btn_t4_eval_browse = QPushButton("📂"); self.btn_t4_eval_browse.clicked.connect(self.browse_tab4_eval_model); h_eval_model.addWidget(self.t4_eval_model_display); h_eval_model.addWidget(self.btn_t4_eval_browse)
        
        f2.addRow("평가 대상 모델:", h_eval_model); self.add_param(f2, "Confidence Threshold", self.t4_conf, d['eval_conf']); self.add_param(f2, "IoU (NMS)", self.t4_iou, d['eval_iou']); self.add_param(f2, "정답 매칭 IoU", self.t4_match_iou, d['eval_match']); self.add_param(f2, "최대 탐지 수", self.t4_max_det, d['eval_max']); self.add_param(f2, "Agnostic NMS", self.t4_agnostic, d['eval_agnostic']); eval_layout.addLayout(f2)
        
        h_eval_btns = QHBoxLayout()
        self.t4_btn_eval = QPushButton("📊 재학습 모델 최종 평가 실행"); self.t4_btn_eval.setMinimumHeight(40); self.t4_btn_eval.clicked.connect(self.run_tab4_eval); self.t4_btn_eval.setEnabled(False)
        self.t4_btn_auto_thr = QPushButton("🎯 스레숄드 찾기"); self.t4_btn_auto_thr.setMinimumHeight(40); self.t4_btn_auto_thr.setStyleSheet("background-color: #fef08a; font-weight: bold; color: #854d0e;"); self.t4_btn_auto_thr.clicked.connect(self.run_tab4_auto_threshold)
        self.t4_btn_reset = QPushButton("🔄 이 탭 초기화"); self.t4_btn_reset.setMinimumHeight(40); self.t4_btn_reset.setStyleSheet("background-color: #fee2e2; border: 1px solid #fca5a5; font-weight: bold; color: #b91c1c;")
        self.t4_btn_reset.clicked.connect(lambda _, w=left_widget: self.reset_tab_defaults(w, "재학습"))
        
        h_eval_btns.addWidget(self.t4_btn_eval); h_eval_btns.addWidget(self.t4_btn_auto_thr); h_eval_btns.addWidget(self.t4_btn_reset)
        eval_layout.addLayout(h_eval_btns)
        
        self.btn_send_t4_to_t5 = QPushButton("➡️ 이 모델과 설정으로 거리 측정"); self.btn_send_t4_to_t5.setStyleSheet("background-color: #dbeafe; font-weight: bold; color: #1e3a8a;"); self.btn_send_t4_to_t5.clicked.connect(lambda: self.send_to_measure_tab(self.t4_eval_model_display.text(), self.t3_img.get_path(), self.t4_conf.value(), self.t4_iou.value(), self.t4_max_det.value(), self.t4_agnostic.isChecked())); eval_layout.addWidget(self.btn_send_t4_to_t5); st2_layout.addWidget(eval_group)
        self.t4_table = QTableWidget(0, 5); self.t4_table.setHorizontalHeaderLabels(["파일명", "상태", "예 예측 수", "정답 수", "사유"]); self._apply_table_style(self.t4_table); header = self.t4_table.horizontalHeader()
        for i in range(self.t4_table.columnCount()): header.setSectionResizeMode(i, QHeaderView.Stretch)
        self.t4_table.setSortingEnabled(True); self.t4_table.itemDoubleClicked.connect(self.on_t4_table_double_clicked); st2_layout.addWidget(QLabel("<b>전체 데이터 최종 평가 결과</b>")); st2_layout.addWidget(self.t4_table)
        sub_tabs.addTab(sub_tab1, "⚙️ 1. 재학습 설정 및 실행"); sub_tabs.addTab(sub_tab2, "📊 2. 재학습 모델 최종 평가"); main_split.addWidget(left_widget)
        
        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget); right_layout.setContentsMargins(10, 0, 0, 0); self.t4_chk_show_all = QCheckBox("전체 평가 이미지 보기"); self.t4_chk_show_all.setEnabled(False); self.t4_chk_show_all.stateChanged.connect(self.update_tab4_visualization)
        r_header = QHBoxLayout(); r_header.addWidget(QLabel("<b>최종 예측 결과 시각화</b>")); r_header.addStretch(1); r_header.addWidget(self.t4_chk_show_all); self.t4_img_grid = ImageGridWidget(max_display=100); right_layout.addLayout(r_header); right_layout.addWidget(self.t4_img_grid)
        main_split.addWidget(right_widget); main_split.setSizes([600, 800]); tab_layout = QVBoxLayout(); tab_layout.addWidget(main_split); tab = QWidget(); tab.setLayout(tab_layout); self.tabs.addTab(tab, "🔁 재학습")

    def run_tab4_auto_threshold(self):
        logger.info("재학습 모델 최적 스레숄드 자동 탐색 실행 시작")
        if self.training_process and self.training_process.is_alive(): 
            QMessageBox.warning(self, "경고", "이미 진행 중입니다."); return
        
        model_path = self.t4_eval_model_display.text()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "경고", "먼저 재학습 모델을 불러오거나 재학습을 완료해주세요."); return
            
        is_valid, err = self.validate_paths(평가이미지_dir=Path(self.t3_img.get_path()), 정답라벨_dir=Path(self.t3_lbl.get_path()))
        if not is_valid: QMessageBox.warning(self, "경로 오류", err); return
        
        msg = ('설정된 [정답 매칭 IoU] 기준에 맞춰, 재학습된 모델의 성능을 최대로 만드는\n'
               '최적의 Confidence와 NMS IoU 값을 자동 탐색합니다.\n\n'
               '💡 탐색 로직 안내:\n'
               '- 탐색 목표: 1순위(완벽 정답 이미지 수 최대화), 2순위(F1-Score 최대화)\n'
               '- 탐색 범위: Conf(0.01~0.99), NMS IoU(0.15~0.85)\n'
               '- 방식: 1차 넓은 범위 탐색 후 최적점 근처 2차 정밀 탐색\n\n'
               '주의: 평가용 데이터셋이 너무 적으면 오버피팅될 수 있으므로 주의하세요.\n\n'
               '진행하시겠습니까?')
               
        if QMessageBox.question(self, '스레숄드 탐색', msg, QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
        
        args = {"model_path": model_path, "img_dir": self.t3_img.get_path(), "lbl_dir": self.t3_lbl.get_path(), "match_iou": self.t4_match_iou.value(), "agnostic": self.t4_agnostic.isChecked(), "max_det": self.t4_max_det.value()}
        self.start_training_process(_auto_threshold_worker, args)

    def run_tab4_retrain(self):
        logger.info("Tab4 재학습 실행 버튼 클릭")
        if self.training_process and self.training_process.is_alive(): QMessageBox.warning(self, "경고", "이미 진행 중"); return
        is_valid, err = self.validate_paths(오답_dir=Path(self.t4_hard.get_path()), 원본정답_dir=Path(self.t4_orig.get_path()), 베이스모델_file=Path(self.t4_base.get_path()))
        if not is_valid: QMessageBox.warning(self, "경로 오류", err); return
        success, cmap_or_error = self.parse_and_validate_class_map(self.t1_class_map.toPlainText())
        if not success: QMessageBox.warning(self, "오류", cmap_or_error); return
        args = {"rt_hard_dir": self.t4_hard.get_path(), "rt_orig_labels": self.t4_orig.get_path(), "rt_base_model": self.t4_base.get_path(), "workspace_dir": self.w_work_ds.get_path(), "webhook_url": self.webhook_url, "noti_flags": self.get_noti_flags(), "imgsz": int(self.g_imgsz.currentText()), "class_names": list(cmap_or_error.keys()), "rt_epochs": self.t4_epochs.value(), "rt_batch": self.t4_batch.value(), "rt_run_name": self.t4_run.text(), "rt_cls": self.t4_lcls.value(), "rt_box": self.t4_lbox.value(), "rt_flipud": self.t4_afud.value(), "rt_fliplr": self.t4_aflr.value(), "rt_mosaic": self.t4_amos.value(), "rt_h": self.t4_ah.value(), "rt_s": self.t4_as.value(), "rt_v": self.t4_av.value(), "rt_mix": self.t4_amix.value(), "rt_cp": self.t4_acp.value(), "eval_img_dir": self.t3_img.get_path(), "eval_lbl_dir": self.t3_lbl.get_path(), "match_iou": getattr(self, 't3_match_iou', QDoubleSpinBox()).value()}
        self.t4_btn_eval.setEnabled(False); self.start_training_process(_retrain_worker, args)

    def run_tab4_eval(self):
        logger.info("Tab4 최종 평가 버튼 클릭")
        from ultralytics import YOLO; import yaml
        is_valid, err = self.validate_paths(평가모델_file=Path(self.t4_eval_model_display.text()), 평가이미지_dir=Path(self.t3_img.get_path()), 정답라벨_dir=Path(self.t3_lbl.get_path()))
        if not is_valid: QMessageBox.warning(self, "경로 오류", err); return
        model_to_eval, eval_img, eval_lbl = Path(self.t4_eval_model_display.text()), Path(self.t3_img.get_path()), Path(self.t3_lbl.get_path())
        self.statusBar().showMessage("🔍 평가 준비 중..."); QApplication.setOverrideCursor(Qt.WaitCursor); eval_yaml_dir = Path(self.w_work_ds.get_path()) / "eval_tmp"
        try:
            temp_model = YOLO(str(model_to_eval)); model_class_names = temp_model.names; num_classes = len(model_class_names); class_names_list = [model_class_names[i] for i in range(num_classes)]; del temp_model; clear_vram()
            eval_yaml_dir.mkdir(parents=True, exist_ok=True); yaml_path = eval_yaml_dir / "eval_data.yaml"
            yaml_path.write_text(yaml.dump({"path": eval_img.parent.resolve().as_posix(), "train": eval_img.name, "val": eval_img.name, "nc": num_classes, "names": class_names_list}, sort_keys=False), encoding="utf-8")
        except Exception as e: 
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "오류", f"정보 추출 실패:\n{e}")
            return
            
        config = {"retrained_model": str(model_to_eval), "yaml_path": str(yaml_path), "eval_source": str(eval_img), "gt_labels_path": str(eval_lbl), "workspace_dir": self.w_work_ds.get_path(), "eval_run_name": self.t4_run.text(), "eval_conf": self.t4_conf.value(), "eval_iou": self.t4_iou.value(), "match_iou": self.t4_match_iou.value(), "max_det": self.t4_max_det.value(), "agnostic_nms": self.t4_agnostic.isChecked(), "webhook_url": self.webhook_url, "noti_flags": self.get_noti_flags()}
        
        if self.webhook_url and self.noti_flags.get("start"): 
            send_discord_webhook(
                webhook_url=self.webhook_url,
                title="▶️ [작업 시작]",
                description="재학습 모델 최종 평가 시작",
                color=0x9b59b6
            )
        
        self.t4_btn_eval.setEnabled(False); self.statusBar().showMessage(f"📊 재학습 평가 진행 중...")
        self.t4_thread = Tab4FinalEvalThread(config)
        self.t4_thread.finished_ok.connect(self.on_tab4_eval_finished)
        self.t4_thread.error.connect(lambda e: self.on_thread_error("재학습 모델 평가", e))
        
        def cleanup_tmp():
            if eval_yaml_dir.exists(): shutil.rmtree(eval_yaml_dir, ignore_errors=True)
            
        self.t4_thread.finished.connect(lambda: [self.t4_btn_eval.setEnabled(True), QApplication.restoreOverrideCursor(), self.statusBar().clearMessage(), cleanup_tmp()])
        self.t4_thread.start()

    def on_tab4_eval_finished(self, df, wrong_imgs, all_imgs, stats, pr_curve):
        self.t4_last_wrong_imgs = wrong_imgs; self.t4_last_all_imgs = all_imgs; self.t4_table.setSortingEnabled(False); self.t4_table.setRowCount(len(df))     
        for i, row in df.iterrows():
            self.t4_table.setItem(i, 0, QTableWidgetItem(str(row["파일명"]))); self.t4_table.setItem(i, 1, QTableWidgetItem(str(row["상태"]))); self.t4_table.setItem(i, 2, QTableWidgetItem(str(row["예 예측 수"]))); self.t4_table.setItem(i, 3, QTableWidgetItem(str(row["정답 수"]))); self.t4_table.setItem(i, 4, QTableWidgetItem(str(row["사유"])))
            for col in range(5): self.t4_table.item(i, col).setTextAlignment(Qt.AlignCenter)
        self.t4_table.setSortingEnabled(True); self.t4_chk_show_all.setEnabled(True); self.t4_chk_show_all.blockSignals(True); self.t4_chk_show_all.setChecked(False); self.t4_chk_show_all.blockSignals(False); self.update_tab4_visualization()
        
        eval_path = self.t4_eval_model_display.text()
        display_name = Path(eval_path).name
        if display_name == "best_model.pt":
            source_txt = Path(eval_path).parent / "best_model_source.txt"
            if source_txt.exists():
                display_name += f" (원본: {source_txt.read_text(encoding='utf-8').strip()})"
                
        model_name_with_path = f"{display_name}  |  {eval_path}"
        self.log_db.insert_eval_log(task_type="Tab 4 Final Eval", model_name=model_name_with_path, total=stats['total'], wrong=stats['wrong'], accuracy=stats['acc'], wrong_imgs_list=wrong_imgs, config_data=self.config_builder.build(self))
        
        if self.webhook_url and self.noti_flags.get("task"):
            send_discord_webhook(
                webhook_url=self.webhook_url,
                title="✅ [작업 완료] 재학습 모델 최종 평가 완료",
                description=f"전체 {stats['total']}장 중 {stats['wrong']}장 오답 (정확도 {stats['acc']:.1f}%)",
                color=0x2ecc71
            )
        
        QMessageBox.information(self, "완료", f"총 {stats['total']}장 중 {stats['correct']}장 완벽 일치, {stats['wrong']}장 이상 발생\n\n[객체 단위 검출 성능]\n- 정밀도(Precision): {stats.get('precision', 0):.1f}%\n- 재현율(Recall): {stats.get('recall', 0):.1f}%\n- F1-Score: {stats.get('f1_score', stats['acc']):.1f}%")

    def update_tab4_visualization(self):
        if hasattr(self, 't4_last_all_imgs'): self.t4_img_grid.update_images(self.t4_last_all_imgs if self.t4_chk_show_all.isChecked() else self.t4_last_wrong_imgs)

    def on_t4_table_double_clicked(self, item):
        if hasattr(self, 't4_last_all_imgs'):
            try: index = next(i for i, path in enumerate(self.t4_last_all_imgs) if Path(path).name == self.t4_table.item(item.row(), 0).text()); ImagePreviewDialog(self.t4_last_all_imgs, index, self).exec_()
            except StopIteration: QMessageBox.warning(self, "오류", "파일을 찾을 수 없습니다.")

    def setup_tab5(self):
        d = ConfigDefaults.TAB5
        f = QFormLayout(); proc_dir, work_dir = Path(self.w_proc_ds.get_path()), Path(self.w_work_ds.get_path())
        
        self.t5_model = PathInputWidget("거리 측정 모델", False, str(work_dir/"kfold"/"best_model.pt")); f.addRow(self.t5_model); self.t5_img = PathInputWidget("분석할 이미지 폴더", True, str(proc_dir/"images")); f.addRow(self.t5_img)
        
        self.t5_method = QComboBox(); self.t5_method.addItems(["테두리 최단거리 (Edge)", "중심점 유클리드 (Center)", "가장 가까운 N개 이웃 (방향 무관)"])
        self.t5_conf = QDoubleSpinBox(); self.t5_conf.setRange(0.01, 0.99); self.t5_conf.setValue(d['conf']); self.t5_conf.setSingleStep(0.01)
        self.t5_iou = QDoubleSpinBox(); self.t5_iou.setRange(0.01, 0.99); self.t5_iou.setValue(d['iou']); self.t5_iou.setSingleStep(0.05)
        self.t5_max_det = QSpinBox(); self.t5_max_det.setRange(1, 1000); self.t5_max_det.setValue(d['max_det'])
        self.t5_agnostic = QCheckBox(); self.t5_agnostic.setChecked(d['agnostic'])
        self.t5_knn_n = QSpinBox(); self.t5_knn_n.setRange(1, 10); self.t5_knn_n.setValue(d['knn_n'])
        self.t5_edge_thr = QLineEdit(d['edge_thr'])
        self.t5_skip_ng = QCheckBox(); self.t5_skip_ng.setChecked(d['skip_ng'])
        self.t5_drop_odd = QCheckBox(); self.t5_drop_odd.setChecked(d['drop_odd'])
        
        color_map = {"노란색 (Yellow)": QColor(255, 255, 0), "초록색 (Green)": QColor(0, 255, 0), "빨간색 (Red)": QColor(255, 0, 0), "파란색 (Blue)": QColor(0, 0, 255), "청록색 (Cyan)": QColor(0, 255, 255), "자주색 (Magenta)": QColor(255, 0, 255), "흰색 (White)": QColor(255, 255, 255)}
        
        def get_color_icon(color):
            pixmap = QPixmap(16, 16); pixmap.fill(Qt.gray); painter = QPainter(pixmap); painter.fillRect(1, 1, 14, 14, color); painter.end(); return QIcon(pixmap)
            
        self.t5_color1 = QComboBox(); self.t5_color2 = QComboBox()
        for name, color in color_map.items(): self.t5_color1.addItem(get_color_icon(color), name); self.t5_color2.addItem(get_color_icon(color), name)
        
        self.t5_color1.setCurrentText(d['color1']); self.t5_color2.setCurrentText(d['color2'])
        
        self.add_param(f, "측정 방식", self.t5_method, d['method']); self.add_param(f, "Confidence", self.t5_conf, d['conf']); self.add_param(f, "IoU (NMS)", self.t5_iou, d['iou']); self.add_param(f, "최대 탐지 수", self.t5_max_det, d['max_det']); self.add_param(f, "Agnostic NMS", self.t5_agnostic, d['agnostic']); self.add_param(f, "N개 이웃 (KNN 전용)", self.t5_knn_n, d['knn_n']); self.add_param(f, "Edge 분류 임계값", self.t5_edge_thr, d['edge_thr']); self.add_param(f, "'NG' 클래스 제외", self.t5_skip_ng, d['skip_ng']); self.add_param(f, "홀수 개체 시 최저 신뢰도 제거", self.t5_drop_odd, d['drop_odd']); self.add_param(f, "수평/KNN 선 색상", self.t5_color1, d['color1']); self.add_param(f, "수직 선 색상 (Edge)", self.t5_color2, d['color2'])
        
        self.t5_btn_run = QPushButton("📏 측정 및 통계 분석 실행"); self.t5_btn_run.clicked.connect(self.run_tab5)
        self.t5_progress = QProgressBar(); self.t5_canvas = FigureCanvas(plt.Figure(figsize=(10, 4.5))); self.t5_canvas.setMinimumHeight(300); self.t5_toolbar = NavigationToolbar(self.t5_canvas, self); self.t5_img_grid = ImageGridWidget(max_display=100)
        self.t5_chk_show_outliers = QCheckBox("🚨 이상치(Outlier) 이미지만 필터링"); self.t5_chk_show_outliers.setStyleSheet("color: red; font-weight: bold;"); self.t5_chk_show_outliers.setEnabled(False); self.t5_chk_show_outliers.stateChanged.connect(self.update_tab5_visualization)
        
        split = QSplitter(Qt.Horizontal); c_w = QWidget(); c_l = QVBoxLayout(c_w); c_l.addWidget(self._create_scroll(f), stretch=1)
        
        self.t5_btn_reset = QPushButton("🔄 이 탭 초기화"); self.t5_btn_reset.setStyleSheet("background-color: #fee2e2; border: 1px solid #fca5a5; padding: 5px; border-radius: 4px;")
        self.t5_btn_reset.clicked.connect(lambda _, w=c_w: self.reset_tab_defaults(w, "거리 측정"))
        h_t5_btns = QHBoxLayout(); h_t5_btns.addWidget(self.t5_btn_run); h_t5_btns.addWidget(self.t5_btn_reset)
        
        c_l.addLayout(h_t5_btns); c_l.addWidget(self.t5_progress); c_l.addWidget(QLabel("<b>데이터 분포 시각화</b>")); c_l.addWidget(self.t5_toolbar); c_l.addWidget(self.t5_canvas, stretch=2)
        i_w = QWidget(); i_l = QVBoxLayout(i_w); i_header_layout = QHBoxLayout(); i_header_layout.addWidget(QLabel("<b>측정 결과 시각화</b>")); i_header_layout.addStretch(1); i_header_layout.addWidget(self.t5_chk_show_outliers); i_l.addLayout(i_header_layout); i_l.addWidget(self.t5_img_grid)
        split.addWidget(c_w); split.addWidget(i_w); split.setSizes([700, 700]); l = QVBoxLayout(); l.addWidget(split); tab = QWidget(); tab.setLayout(l); self.tabs.addTab(tab, "📏 거리 측정")

    def run_tab5(self):
        logger.info("Tab5 거리 측정 실행 버튼 클릭")
        is_valid, err = self.validate_paths(측정모델_file=Path(self.t5_model.get_path()), 분석이미지_dir=Path(self.t5_img.get_path()))
        if not is_valid: QMessageBox.warning(self, "오류", err); return
        config = {"dist_model_path": self.t5_model.get_path(), "dist_source": self.t5_img.get_path(), "workspace_dir": self.w_work_ds.get_path(), "measure_method": self.t5_method.currentText(), "dist_conf": self.t5_conf.value(), "dist_iou": self.t5_iou.value(), "dist_max_det": self.t5_max_det.value(), "dist_agnostic": self.t5_agnostic.isChecked(), "n_neighbors": self.t5_knn_n.value(), "edge_thresholds": self.t5_edge_thr.text(), "skip_ng": self.t5_skip_ng.isChecked(), "drop_odd_lowest": self.t5_drop_odd.isChecked(), "color1": self.t5_color1.currentText(), "color2": self.t5_color2.currentText(), "webhook_url": self.webhook_url, "noti_flags": self.get_noti_flags()}
        
        if self.webhook_url and self.noti_flags.get("start"): 
            send_discord_webhook(
                webhook_url=self.webhook_url,
                title="▶️ [작업 시작]",
                description="객체 간 거리 측정 및 통계 분석 시작",
                color=0x9b59b6
            )
        
        self.t5_btn_run.setEnabled(False); self.statusBar().showMessage("📏 측정 진행 중..."); QApplication.setOverrideCursor(Qt.WaitCursor); self.t5_progress.setValue(0)
        self.t5_thread = MeasureThread(config); self.t5_thread.progress.connect(self.t5_progress.setValue); self.t5_thread.finished_ok.connect(self.on_tab5_finished); self.t5_thread.error.connect(lambda e: self.on_thread_error("거리 측정", e))
        self.t5_thread.finished.connect(lambda: [self.t5_btn_run.setEnabled(True), QApplication.restoreOverrideCursor(), self.statusBar().clearMessage()]); self.t5_thread.start()

    def on_tab5_finished(self, df_export, df_parsed, df_outliers, image_pairs):
        msg = f"완료! (총 {len(df_export)}건)\n\n📂 저장 목록:\n - results.csv\n - statistics.csv\n"
        if not df_outliers.empty: msg += f" - outliers.csv (🚨 {len(df_outliers)}건)\n"
        
        if self.webhook_url and self.noti_flags.get("task"):
            send_discord_webhook(
                webhook_url=self.webhook_url,
                title="✅ [작업 완료] 거리 측정",
                description=f"거리 측정 및 분석 완료 (총 {len(df_export)}건)",
                color=0x2ecc71
            )
        
        QMessageBox.information(self, "완료", msg); self.t5_last_df_outliers = df_outliers; self.t5_last_image_pairs = image_pairs
        self.t5_chk_show_outliers.setEnabled(True); self.t5_chk_show_outliers.blockSignals(True); self.t5_chk_show_outliers.setChecked(False); self.t5_chk_show_outliers.blockSignals(False)
        if not df_parsed.empty:
            self.t5_canvas.figure.clear(); ax1 = self.t5_canvas.figure.add_subplot(121); ax2 = self.t5_canvas.figure.add_subplot(122)
            sns.kdeplot(data=df_parsed, x="거리(px)", hue="구분", fill=True, ax=ax1, palette="Set1", alpha=0.5); ax1.set_title("KDE Plot")
            sns.boxenplot(data=df_parsed, x="구분", y="거리(px)", ax=ax2, palette="Set1"); ax2.set_title("Boxen Plot")
            if not df_outliers.empty: sns.scatterplot(data=df_outliers, x="구분", y="거리(px)", color="red", marker="X", s=100, ax=ax2, label="Outliers", zorder=5)
            self.t5_canvas.figure.tight_layout(); self.t5_canvas.draw()
        self.update_tab5_visualization()

    def update_tab5_visualization(self):
        if not hasattr(self, 't5_last_image_pairs'): return
        if self.t5_chk_show_outliers.isChecked():
            if self.t5_last_df_outliers.empty: self.t5_img_grid.update_images([]) 
            else:
                outlier_files = set(self.t5_last_df_outliers["파일명"].tolist())
                self.t5_img_grid.update_images([item["_img_path"] for item in self.t5_last_image_pairs if item["파일명"] in outlier_files])
        else: self.t5_img_grid.update_images([item["_img_path"] for item in self.t5_last_image_pairs])
if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    app = QApplication(sys.argv)
    font_name = "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"
    app.setFont(QFont(font_name, 10))
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    
    window = MainWindow()
    
    # 🚨 글로벌 에러 핸들러 (프로그램이 튕기기 직전에 낚아채서 웹훅 발송)
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        # 설정에 웹훅이 켜져있다면 유언(웹훅) 전송
        if hasattr(window, 'webhook_url') and window.webhook_url:
            error_details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            # 메시지가 너무 길면 디스코드 제한(2000자)에 걸리므로 자르기
            if len(error_details) > 1000:
                error_details = error_details[-1000:] 
                
            send_discord_webhook(
                webhook_url=window.webhook_url,
                title="💀 [프로세스 치명적 충돌]",
                description=f"UI 메인 프로그램에서 처리되지 않은 에러가 발생하여 종료됩니다.\n```{error_details}```",
                color=0x992d22, # 다크 레드
                sync=True       # 확실하게 전송될 때까지 대기
            )
            
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    # 파이썬 기본 에러 핸들러를 우리가 만든 핸들러로 덮어치기
    sys.excepthook = global_exception_handler
    
    window.showMaximized()
    sys.exit(app.exec_())