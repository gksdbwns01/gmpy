from ...utils import *
from ...core import *
from ...core.workers import *
from ..ui_components import *


class MeasureTabMixin:
    def send_to_measure_tab(self, model_path, img_path, conf, iou, max_det, agnostic):
        if not model_path or not Path(model_path).exists(): QMessageBox.warning(self, "경로 오류", "유효한 모델 경로가 없습니다."); return
        self.t5_model.line_edit.setText(model_path); self.t5_img.line_edit.setText(img_path); self.t5_conf.setValue(conf); self.t5_iou.setValue(iou); self.t5_max_det.setValue(max_det); self.t5_agnostic.setChecked(agnostic); self.tabs.setCurrentIndex(5); self.statusBar().showMessage("➡️ 현재 모델, 이미지 경로, 평가 설정이 거리 측정 탭으로 복사되었습니다.", 5000)
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
