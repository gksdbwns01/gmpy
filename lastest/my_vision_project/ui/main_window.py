from ..utils import *
from ..core import *
from ..core.workers import *
from .ui_components import *
from .tabs.tab_eval import EvalTabMixin
from .tabs.tab_labeling import LabelingTabMixin
from .tabs.tab_measure import MeasureTabMixin
from .tabs.tab_preprocess import PreprocessTabMixin
from .tabs.tab_project import ProjectTabMixin
from .tabs.tab_retrain import RetrainTabMixin
from .tabs.tab_train import TrainTabMixin

class MainWindow(QMainWindow, LabelingTabMixin, ProjectTabMixin, PreprocessTabMixin, TrainTabMixin, EvalTabMixin, RetrainTabMixin, MeasureTabMixin):
    def __init__(self):
        super().__init__(); self.setWindowTitle(APP_WINDOW_TITLE); self.resize(1400, 900)
        logger.info(f"YOLO Training Pipeline 애플리케이션 시작 (OS: {platform.system()}, GPU: {torch.cuda.is_available()})")
        self.base_dir = Path(sys.executable).parent if getattr(sys, 'frozen', False) else get_app_base_dir()
        self.settings = QSettings(APP_ORG_NAME, APP_NAME)
        last_proj = self.settings.value("last_project_path", str(self.base_dir))
        self.config_manager = ConfigManager(last_proj)
        self.config_builder = ConfigBuilder()
        self.log_db = LogDatabase(self.base_dir / "logs" / "training_history.db")
        self.training_process = None
        self.init_ui()
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status_animation)
        self.status_animation_frame = 0
        self.base_status_msg = ""

    def update_status_animation(self):
        frames = ["⏳", "⌛"]
        frame = frames[self.status_animation_frame % len(frames)]
        dots = "." * ((self.status_animation_frame % 3) + 1)
        self.status_animation_frame += 1
        pid_info = f"[PID: {self.training_process.pid}] " if self.training_process and self.training_process.is_alive() else ""
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
                if path.stat().st_size > MODEL_MAX_SIZE_BYTES:
                    return False, f"[{name}] 파일 용량이 너무 큽니다 (5GB 제한):\n{path}"
                    
            elif name.endswith('_dir') or "ds" in name.lower() or "root" in name.lower():
                if not path.is_dir():
                    return False, f"[{name}] 폴더(디렉토리)여야 합니다:\n{path}"
                    
        return True, None

    def closeEvent(self, event):
        is_busy = (self.training_process and self.training_process.is_alive())
        if is_busy:
            reply = QMessageBox.warning(
                self, "종료 경고", 
                "현재 모델 학습 등 중요 작업이 진행 중입니다.\n지금 프로그램을 강제 종료하면 가중치 파일(.pt)이나 로그 데이터가 영구적으로 손상될 수 있습니다.\n\n먼저 [안전 종료] 버튼을 눌러 작업을 정상적으로 마친 후 창을 닫아주세요.\n\n그래도 무시하고 강제로 종료하시겠습니까?", 
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

        logger.info("애플리케이션 종료 프로세스 시작")
        
        if hasattr(self, 'monitor_thread') and self.monitor_thread:
            try:
                self.monitor_thread.finished_ok.disconnect()
                self.monitor_thread.error.disconnect()
                logger.debug("모니터링 스레드 시그널 연결 해제 (중복 알림 방지)")
            except:
                pass

        if is_busy and self.webhook_url and self.noti_flags.get("task"):
            task_name = "TASK"
            if not self.t2_btn_run.isEnabled(): task_name = "K-FOLD TRAIN / AUTO ML"
            elif not self.t4_btn_retrain.isEnabled(): task_name = "HARD RETRAIN"
            elif not self.t1_btn_run.isEnabled(): task_name = "DATA PREPROCESS"
            elif not self.t3_btn_run.isEnabled(): task_name = "EVALUATION"
            elif not self.t5_btn_run.isEnabled(): task_name = "DISTANCE MEASURE"

            send_discord_webhook(
                webhook_url=self.webhook_url,
                title=f"🛑 [작업 중단] {task_name}",
                description="사용자가 프로그램을 종료하여 진행 중인 작업이 즉시 중단되었습니다.",
                color=0xf39c12, 
                sync=True 
            )

        if self.webhook_url:
            status_text = " (작업 중 종료)" if is_busy else " (정상 종료)"
            send_discord_webhook(
                webhook_url=self.webhook_url,
                title="⏹️ [프로그램 종료]",
                description=f"YOLO 파이프라인 관리 도구가 종료되었습니다.{status_text}",
                color=0x2c3e50,
                sync=True
            )

        if is_busy:
            self.stop_training()
            self.training_process.join(timeout=1.5)
            if self.training_process.is_alive():
                logger.warning("워커 프로세스가 응답하지 않아 강제 종료(Terminate)합니다.")
                kill_process_tree(self.training_process.pid)

        for t_name in ["t1_thread", "t3_thread", "t4_thread", "t5_thread"]:
            if hasattr(self, t_name):
                th = getattr(self, t_name)
                if th and th.isRunning():
                    th.quit()
                    th.wait(1000)

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
        proj_dir = Path(new_path)
        self.w_base_ds.line_edit.setText(str(proj_dir / "dataset"))
        self.w_proc_ds.line_edit.setText(str(proj_dir / "processed_dataset"))
        self.w_work_ds.line_edit.setText(str(proj_dir / "workspace"))
        self.config_manager.update_workspace_path(str(proj_dir / "workspace"))
        self.settings.setValue("last_project_path", new_path)
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
        return getattr(self, "noti_flags", build_default_noti_flags())

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
        
        self.settings = QSettings(APP_ORG_NAME, APP_NAME)
        self.webhook_url = self.settings.value("webhook_url", "")
        self.noti_flags = build_default_noti_flags()
        
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
        if "folds" in t2: self.t2_folds.setValue(t2["folds"])
        if "test_split" in t2: self.t2_test_split.setValue(t2["test_split"])
        if "lcls" in t2: self.t2_lcls.setValue(t2["lcls"])
        if "lbox" in t2: self.t2_lbox.setValue(t2["lbox"])
        if "ldfl" in t2: self.t2_ldfl.setValue(t2["ldfl"])
        if "tune_iterations" in t2: self.t2_tune_iterations.setValue(t2["tune_iterations"])
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
        
        self.monitor_thread = ProcessMonitorThread(self.train_queue, self.training_process, self.stop_event)
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
        
        is_stopped_early = hasattr(self, 'stop_event') and self.stop_event.is_set()

        if res.get("success"):
            if is_stopped_early:
                if self.webhook_url and self.noti_flags.get("task"):
                    send_discord_webhook(
                        webhook_url=self.webhook_url,
                        title=f"🛑 [작업 중단] {task_name}",
                        description="사용자 요청에 의해 작업이 안전하게 종료(Graceful Stop)되었습니다.\n*(중단 전까지 학습된 가중치와 결과 데이터는 정상 보존됩니다.)*",
                        color=0xf39c12
                    )
            else:
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
            error_msg = res.get("error", "알 수 없는 오류")
            
            if "사용자에 의해" in error_msg or "취소" in error_msg:
                if self.webhook_url and self.noti_flags.get("task"):
                    send_discord_webhook(self.webhook_url, f"🛑 [작업 취소] {task_name}", "사용자의 요청으로 작업이 안전하게 취소/중단되었습니다.", color=0xf39c12)
                QMessageBox.information(self, "작업 취소 안내", error_msg)
                
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









































