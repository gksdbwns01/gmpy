from ...utils import *
from ...core import *
from ...core.workers import *
from ..ui_components import *


class RetrainTabMixin:
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
        self.t4_table = QTableWidget(0, 5); self.t4_table.setHorizontalHeaderLabels(["파일명", "상태", "예측 수", "정답 수", "사유"]); self._apply_table_style(self.t4_table); header = self.t4_table.horizontalHeader()
        for i in range(self.t4_table.columnCount()): header.setSectionResizeMode(i, QHeaderView.Stretch)
        self.t4_table.setSortingEnabled(True); self.t4_table.itemDoubleClicked.connect(self.on_t4_table_double_clicked); st2_layout.addWidget(QLabel("<b>전체 데이터 최종 평가 결과</b>")); st2_layout.addWidget(self.t4_table)
        sub_tabs.addTab(sub_tab1, "⚙️ 1. 재학습 설정 및 실행"); sub_tabs.addTab(sub_tab2, "📊 2. 재학습 모델 최종 평가"); main_split.addWidget(left_widget)
        
        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget); right_layout.setContentsMargins(10, 0, 0, 0); self.t4_chk_show_all = QCheckBox("전체 평가 이미지 보기"); self.t4_chk_show_all.setEnabled(False); self.t4_chk_show_all.stateChanged.connect(self.update_tab4_visualization)
        r_header = QHBoxLayout(); r_header.addWidget(QLabel("<b>최종 예측 결과 시각화</b>")); r_header.addStretch(1); r_header.addWidget(self.t4_chk_show_all); self.t4_img_grid = ImageGridWidget(max_display=100); right_layout.addLayout(r_header); right_layout.addWidget(self.t4_img_grid)
        main_split.addWidget(right_widget); main_split.setSizes([600, 800]); tab_layout = QVBoxLayout(); tab_layout.addWidget(main_split); tab = QWidget(); tab.setLayout(tab_layout); self.tabs.addTab(tab, "🔁 재학습")
    def browse_tab4_eval_model(self):
        """Tab 4에서 평가할 재학습 모델(.pt)을 선택하는 탐색창을 엽니다."""
        # 기본 탐색 경로는 워크스페이스(workspace) 폴더로 설정
        default_dir = str(Path(self.w_work_ds.get_path()))
        
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "평가 대상 모델 선택", 
            default_dir, 
            "PyTorch Model (*.pt);;All Files (*)"
        )
        
        if path:
            self.t4_eval_model_display.setText(path)
            # 모델이 선택되었으므로 평가 관련 버튼들 활성화
            self.t4_btn_eval.setEnabled(True)
            self.t4_btn_auto_thr.setEnabled(True)
            self.statusBar().showMessage(f"✅ 평가할 재학습 모델이 선택되었습니다: {Path(path).name}", 5000)
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
        args = {"rt_hard_dir": self.t4_hard.get_path(), "rt_orig_labels": self.t4_orig.get_path(), "rt_base_model": self.t4_base.get_path(), "workspace_dir": self.w_work_ds.get_path(), "webhook_url": self.webhook_url, "noti_flags": self.get_noti_flags(), "imgsz": int(self.g_imgsz.currentText()), "class_names": list(cmap_or_error.keys()), "rt_epochs": self.t4_epochs.value(), "rt_batch": self.t4_batch.value(), "rt_run_name": self.t4_run.text(), "rt_cls": self.t4_lcls.value(), "rt_box": self.t4_lbox.value(), "rt_flipud": self.t4_afud.value(), "rt_fliplr": self.t4_aflr.value(), "rt_mosaic": self.t4_amos.value(), "rt_h": self.t4_ah.value(), "rt_s": self.t4_as.value(), "rt_v": self.t4_av.value(), "rt_mix": self.t4_amix.value(), "rt_cp": self.t4_acp.value(), "eval_img_dir": self.t3_img.get_path(), "eval_lbl_dir": self.t3_lbl.get_path(), "match_iou": getattr(self, 't3_match_iou', QDoubleSpinBox()).value() if hasattr(self, 't3_match_iou') else 0.5}
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
            self.t4_table.setItem(i, 0, QTableWidgetItem(str(row["파일명"]))); self.t4_table.setItem(i, 1, QTableWidgetItem(str(row["상태"]))); self.t4_table.setItem(i, 2, QTableWidgetItem(str(row["예측 수"]))); self.t4_table.setItem(i, 3, QTableWidgetItem(str(row["정답 수"]))); self.t4_table.setItem(i, 4, QTableWidgetItem(str(row["사유"])))
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
