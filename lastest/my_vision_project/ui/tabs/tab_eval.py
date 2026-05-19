from ...utils import *
from ...core import *
from ...core.workers import *
from ..ui_components import *


class EvalTabMixin:
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
