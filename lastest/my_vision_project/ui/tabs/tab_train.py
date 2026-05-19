from ...utils import *
from ...core import *
from ...core.workers import *
from ..ui_components import *


class TrainTabMixin:
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
    def setup_tab2(self):
        d = ConfigDefaults.TAB2
        self.t2_epochs = QSpinBox(); self.t2_epochs.setRange(1, 5000); self.t2_epochs.setValue(d['epochs'])
        self.t2_batch = QSpinBox(); self.t2_batch.setRange(1, 256); self.t2_batch.setValue(d['batch'])
        self.t2_workers = QSpinBox(); self.t2_workers.setRange(0, 32); self.t2_workers.setValue(d['workers'])
        self.t2_patience = QSpinBox(); self.t2_patience.setRange(0, 10000); self.t2_patience.setValue(d['patience'])
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
        self.add_param(f_left, "Epochs", self.t2_epochs, d['epochs']); self.add_param(f_left, "Batch", self.t2_batch, d['batch']); self.add_param(f_left, "Workers", self.t2_workers, d['workers']); self.add_param(f_left, "Patience (조기 종료)", self.t2_patience, d['patience']); self.add_param(f_left, "Fold 수", self.t2_folds, d['folds']); self.add_param(f_left, "Test 분리 비율", self.t2_test_split, d['test_split'])
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
        self.t2_btn_check_integrity = QPushButton("🚨 데이터셋 무결성 검사")
        self.t2_btn_check_integrity.setStyleSheet("background-color: #fee2e2; border: 1px solid #fca5a5; padding: 8px; font-weight: bold; color: #991b1b;")
        self.t2_btn_check_integrity.clicked.connect(self.run_integrity_check)
        h_tune.addWidget(self.t2_btn_tune)
        h_tune.addWidget(self.btn_show_tune_history)
        h_tune.addWidget(self.t2_btn_check_integrity)
        self.t2_btn_run = QPushButton("🚀 K-Fold 학습 시작"); self.t2_btn_run.clicked.connect(self.run_tab2)
        
        self.t2_btn_stop = QPushButton("🛑 안전 종료(Graceful Stop)"); self.t2_btn_stop.clicked.connect(self.stop_training); self.t2_btn_stop.setEnabled(False)
        
        l = QVBoxLayout(); self.t2_scroll = self._create_scroll(h_form); l.addWidget(self.t2_scroll)
        
        self.t2_btn_reset = QPushButton("🔄 이 탭 초기화"); self.t2_btn_reset.setStyleSheet("background-color: #fee2e2; border: 1px solid #fca5a5; padding: 5px; border-radius: 4px;"); self.t2_btn_reset.clicked.connect(lambda _, w=self.t2_scroll: self.reset_tab_defaults(w, "K-Fold 학습"))
        
        l.addLayout(h_tune)
        
        h = QHBoxLayout(); h.addWidget(self.t2_btn_run); h.addWidget(self.t2_btn_stop); h.addWidget(self.t2_btn_reset); l.addLayout(h)
        
        tab = QWidget(); tab.setLayout(l); self.tabs.addTab(tab, "🏋️ K-Fold 학습")
    def run_integrity_check(self):
        logger.info("데이터셋 무결성 검사 실행")
        proc_dir = Path(self.w_proc_ds.get_path())
        is_valid, err = self.validate_paths(처리된_데이터_dir=proc_dir, 이미지_dir=proc_dir/"images", 라벨_dir=proc_dir/"labels")
        if not is_valid: 
            QMessageBox.warning(self, "경로 오류", err)
            return
            
        success, cmap_or_error = self.parse_and_validate_class_map(self.t1_class_map.toPlainText())
        if not success: 
            QMessageBox.warning(self, "클래스 매핑 오류", cmap_or_error)
            return

        self.t2_btn_check_integrity.setEnabled(False)
        self.statusBar().showMessage("🔍 데이터셋 무결성을 검사하는 중입니다...")
        
        from PyQt5.QtWidgets import QProgressDialog
        self.integrity_progress = QProgressDialog("무결성 검사 진행 중...", "취소", 0, 100, self)
        self.integrity_progress.setWindowTitle("Integrity Check")
        self.integrity_progress.setWindowModality(Qt.WindowModal)
        self.integrity_progress.setAutoClose(True)
        
        self.integrity_thread = IntegrityThread(img_dir=proc_dir/"images", lbl_dir=proc_dir/"labels", num_classes=len(cmap_or_error))
        self.integrity_thread.progress.connect(lambda val, msg: (self.integrity_progress.setValue(val), self.integrity_progress.setLabelText(msg)))
        self.integrity_thread.finished_ok.connect(self.on_integrity_finished)
        self.integrity_thread.error.connect(lambda e: self.on_thread_error("무결성 검사", e))
        self.integrity_thread.finished.connect(lambda: self.t2_btn_check_integrity.setEnabled(True))
        
        self.integrity_progress.canceled.connect(self.integrity_thread.terminate)
        self.integrity_thread.start()
    def on_integrity_finished(self, issues):
        self.statusBar().clearMessage()
        if not issues:
            QMessageBox.information(self, "검사 통과", "🎉 발견된 문제가 없습니다! 데이터가 완벽하게 준비되었습니다.")
            return
            
        dialog = IntegrityReportDialog(issues, self)
        dialog.request_fix.connect(self.navigate_to_labeling_for_fix)
        # 🌟 [추가] 새로운 시그널 연결
        dialog.request_fix_multi.connect(self.navigate_to_labeling_for_fix_multi)
        dialog.exec_()
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
            "initial_params": initial_params, "match_iou": getattr(self, 't3_match_iou', QDoubleSpinBox()).value() if hasattr(self, 't3_match_iou') else 0.5, 
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
        args = {"processed_dir": self.w_proc_ds.get_path(), "workspace_dir": self.w_work_ds.get_path(), "webhook_url": self.webhook_url, "noti_flags": self.get_noti_flags(), "model_name": self.g_model.currentText(), "imgsz": int(self.g_imgsz.currentText()), "epochs": self.t2_epochs.value(), "batch": self.t2_batch.value(), "workers": self.t2_workers.value(), "patience": self.t2_patience.value(), "deterministic": False, "num_folds": self.t2_folds.value(), "test_split": self.t2_test_split.value(), "best_metric": "metrics/mAP50-95(B)", "second_metric": "metrics/mAP50(B)", "class_names": list(cmap_or_error.keys()), "aug": {"hsv_h": self.t2_ah.value(), "hsv_s": self.t2_as.value(), "hsv_v": self.t2_av.value(), "degrees": self.t2_adeg.value(), "translate": self.t2_atrans.value(), "scale": self.t2_ascale.value(), "shear": self.t2_ashear.value(), "flipud": self.t2_afud.value(), "fliplr": self.t2_aflr.value(), "mosaic": self.t2_amos.value(), "mixup": self.t2_amix.value(), "copy_paste": self.t2_acp.value()}, "loss": {"cls": self.t2_lcls.value(), "box": self.t2_lbox.value(), "dfl": self.t2_ldfl.value()}, "match_iou": getattr(self, 't3_match_iou', QDoubleSpinBox()).value() if hasattr(self, 't3_match_iou') else 0.5}
        self.start_training_process(_kfold_train_worker, args)
