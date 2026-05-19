from ..utils import *
from ..core import *

class PathInputWidget(QWidget):
    def __init__(self, label, is_folder=True, default_path=""):
        super().__init__(); self.is_folder = is_folder; layout = QHBoxLayout(); layout.setContentsMargins(0, 0, 0, 0)
        self.line_edit = QLineEdit(default_path); self.btn = QPushButton("📂"); self.btn.clicked.connect(self.browse)
        layout.addWidget(QLabel(label)); layout.addWidget(self.line_edit); layout.addWidget(self.btn); self.setLayout(layout)
    def browse(self):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택", self.line_edit.text()) if self.is_folder else QFileDialog.getOpenFileName(self, "파일 선택", self.line_edit.text(), "PyTorch Model (*.pt);;All Files (*)")[0]
        if path: self.line_edit.setText(path)
    def get_path(self): return self.line_edit.text()

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
        self.chk_current_proj_only = QCheckBox("현재 프로젝트 기록만 보기")
        self.chk_current_proj_only.setChecked(True) # 기본값은 현재 프로젝트만 매끄럽게 보여주기
        self.chk_current_proj_only.stateChanged.connect(self.on_search)
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
        filter_layout.addWidget(self.chk_current_proj_only)
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
        current_proj_path = ""
        if self.chk_current_proj_only.isChecked():
            main_window = self.window()
            if hasattr(main_window, 'w_proj_root'):
                current_proj_path = main_window.w_proj_root.get_path()
        offset = (self.current_page - 1) * self.page_size
        rows, total_count = self.db_manager.fetch_logs(
            table_type=self.tab_type, current_proj_path=current_proj_path, 
            search_kw=kw, date_from=d_from, date_to=d_to,
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
        current_proj_path = ""
        if self.chk_current_proj_only.isChecked():
            main_window = self.window()
            if hasattr(main_window, 'w_proj_root'):
                current_proj_path = main_window.w_proj_root.get_path()
        rows, _ = self.db_manager.fetch_logs(
            table_type=self.tab_type, current_proj_path=current_proj_path,
            search_kw=kw, date_from=d_from, date_to=d_to,
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
