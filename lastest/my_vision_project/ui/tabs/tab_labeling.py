from ...utils import *
from ...core import *
from ...core.workers import *
from ..ui_components import *


class LabelingTabMixin:
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
        src_path = Path(src_dir); target_dir = Path(self.t6_img_dir.get_path()); target_dir.mkdir(parents=True, exist_ok=True)
        files_to_copy = iter_supported_images(src_path)
        if not files_to_copy: QMessageBox.warning(self, "파일 없음", "선택한 폴더에 이미지 파일이 없습니다."); return
        copy_count = 0; QApplication.setOverrideCursor(Qt.WaitCursor) 
        try:
            for f in files_to_copy:
                target_file = target_dir / f.name
                if not target_file.exists(): shutil.copy2(f, target_file); copy_count += 1
                
            self.t6_list.clear()
            lbl_dir = Path(self.t6_lbl_dir.get_path()) # 라벨 디렉토리 참조
            
            for f in iter_supported_images(target_dir):
                item = QListWidgetItem(f.name)
                txt_path = lbl_dir / f.with_suffix(".txt").name
                    
                    # 여기서도 하이라이트 검사
                if txt_path.exists() and txt_path.stat().st_size > 0:
                    item.setBackground(QColor("#dcfce7"))
                        
                self.t6_list.addItem(item)
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
    def load_labeling_images_programmatic(self, img_dir_path):
        """다이얼로그 없이 코드로 특정 경로의 이미지를 Tab6 리스트에 강제 로드"""
        target_dir = Path(img_dir_path)
        lbl_dir = Path(self.t6_lbl_dir.get_path()) # 라벨 디렉토리 참조
        self.t6_list.clear()
        
        if target_dir.exists():
            for f in sorted(target_dir.iterdir()):
                if is_supported_image(f):
                    item = QListWidgetItem(f.name)
                    txt_path = lbl_dir / f.with_suffix(".txt").name
                    
                    # 라벨 파일이 존재하고, 내용이 비어있지 않다면 배경색 적용
                    if txt_path.exists() and txt_path.stat().st_size > 0:
                        item.setBackground(QColor("#dcfce7"))
                        
                    self.t6_list.addItem(item)
    def navigate_to_labeling_for_fix(self, file_name):
        logger.info(f"오류 수정을 위해 라벨링 툴로 자동 이동 요청됨: {file_name}")
        target_stem = Path(file_name).stem
        self.tabs.setCurrentIndex(0)
        proc_dir = Path(self.w_proc_ds.get_path())
        self.t6_img_dir.line_edit.setText(str(proc_dir / "images"))
        self.t6_lbl_dir.line_edit.setText(str(proc_dir / "labels"))
        self.load_labeling_images_programmatic(proc_dir / "images")
        
        items = self.t6_list.findItems(file_name, Qt.MatchExactly)
        if not items:
            items = self.t6_list.findItems(target_stem, Qt.MatchContains)
            
        if items:
            # 1. 목록에서 해당 아이템을 선택 및 활성화
            self.t6_list.setCurrentItem(items[0])
            # 2. 🌟 목록 스크롤바가 타겟 위치로 자동 추적하도록 설정
            self.t6_list.scrollToItem(items[0], QListWidget.EnsureVisible) 
            
            # 3. 🌟 [핵심] 실제 이미지와 라벨링 하이라이트 박스들을 즉시 그리도록 직접 호출
            self.on_label_image_selected(items[0])
            
            # 4. 키보드 단축키(A, D, S, W) 조작이 즉시 가능하도록 뷰어 포커스 지정
            self.t6_view.setFocus()
            self.statusBar().showMessage(f"🛠️ '{file_name}' 데이터를 수정할 준비가 되었습니다.", 5000)
        else:
            QMessageBox.warning(self, "파일 탐색 실패", f"라벨링 목록에서 '{file_name}' 이미지를 찾을 수 없습니다.")
    def navigate_to_labeling_for_fix_multi(self, file_names):
        """다수의 오류 이미지를 라벨링 탭 리스트에 한 번에 로드합니다."""
        logger.info(f"오류 수정을 위해 라벨링 툴로 다중 이동 요청됨: {len(file_names)}건")
        self.tabs.setCurrentIndex(0)
        proc_dir = Path(self.w_proc_ds.get_path())
        self.t6_img_dir.line_edit.setText(str(proc_dir / "images"))
        self.t6_lbl_dir.line_edit.setText(str(proc_dir / "labels"))
        
        self.t6_list.clear()
        target_dir = proc_dir / "images"
        lbl_dir = proc_dir / "labels" # 라벨 디렉토리 참조
        
        target_stems = {Path(name).stem for name in file_names}
        
        if target_dir.exists():
            for f in sorted(target_dir.iterdir()):
                if is_supported_image(f) and f.stem in target_stems:
                    item = QListWidgetItem(f.name)
                    txt_path = lbl_dir / f.with_suffix(".txt").name
                    
                    # 라벨 파일이 존재하고, 내용이 비어있지 않다면 배경색 적용
                    if txt_path.exists() and txt_path.stat().st_size > 0:
                        item.setBackground(QColor("#dcfce7"))
                        
                    self.t6_list.addItem(item)
        
        if self.t6_list.count() > 0:
            self.t6_list.setCurrentRow(0)
            first_item = self.t6_list.item(0)
            self.on_label_image_selected(first_item)
            self.t6_view.setFocus()
            self.statusBar().showMessage(f"🛠️ {self.t6_list.count()}개의 문제 이미지를 라벨링 툴에 로드했습니다. (순차적으로 라벨링을 진행하세요)", 6000)
        else:
            QMessageBox.warning(self, "파일 탐색 실패", "지정된 오류 이미지들을 찾을 수 없습니다. (이미 삭제되었을 수 있습니다)")
