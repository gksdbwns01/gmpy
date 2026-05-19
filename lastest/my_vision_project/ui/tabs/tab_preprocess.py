from ...utils import *
from ...core import *
from ...core.workers import *
from ..ui_components import *


class PreprocessTabMixin:
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
        
        self.t1_btn_reset = QPushButton("🔄 이 탭 초기화"); self.t1_btn_reset.setStyleSheet("background-color: #fee2e2; border: 1px solid #fca5a5; padding: 5px; border-radius: 4px;")
        self.t1_btn_reset.clicked.connect(lambda _, w=left_w: self.reset_tab_defaults(w, "데이터 전처리"))
        h_t1_btns = QHBoxLayout(); h_t1_btns.addWidget(self.t1_btn_run); h_t1_btns.addWidget(self.t1_btn_reset)
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
        if target_dir.exists(): self.t1_img_grid.update_images([str(f) for f in iter_supported_images(target_dir)])
