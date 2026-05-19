from ..utils import *
from ..core import *

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

# ==========================================
# 기존 UI 클래스들 (LabelDialog, LabelingView 등)

class LabelDialog(QDialog):
    def __init__(self, class_names, parent=None):
        super().__init__(parent); self.setWindowTitle("클래스 선택"); self.class_id = -1
        layout = QVBoxLayout(self); self.combo = QComboBox(); self.combo.addItems(class_names)
        layout.addWidget(QLabel("객체의 클래스를 선택하세요:")); layout.addWidget(self.combo)
        btn_layout = QHBoxLayout(); btn_ok = QPushButton("확인"); btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("취소"); btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_ok); btn_layout.addWidget(btn_cancel); layout.addLayout(btn_layout)
    def get_class_index(self): return self.combo.currentIndex()

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

class IntegrityReportDialog(QDialog):
    request_fix = pyqtSignal(str) # 기존 단일 이동 시그널
    request_fix_multi = pyqtSignal(list) # 🌟 [추가] 다중 이동을 위한 새로운 시그널

    def __init__(self, issues, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🚨 데이터셋 무결성 검사 리포트")
        self.resize(800, 500)
        self.issues = issues # 🌟 [추가] issues 데이터를 클래스 변수로 저장
        
        layout = QVBoxLayout(self)
        
        lbl_info = QLabel(f"<b>총 {len(issues)}개의 잠재적 문제</b>가 발견되었습니다.<br>"
                          "<span style='color: #d97706;'>💡 항목을 <b>더블클릭</b>하면 해당 이미지만 라벨링 툴에서 엽니다.</span>")
        layout.addWidget(lbl_info)
        
        self.table = QTableWidget(len(issues), 3)
        self.table.setHorizontalHeaderLabels(["파일명", "오류 유형", "상세 설명"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        
        for i, issue in enumerate(issues):
            self.table.setItem(i, 0, QTableWidgetItem(issue["file"]))
            
            type_item = QTableWidgetItem(issue["type"])
            type_item.setForeground(QColor("#dc2626") if "오류" in issue["type"] or "누락" in issue["type"] or "손상" in issue["type"] else QColor("#b45309"))
            type_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.table.setItem(i, 1, type_item)
            
            self.table.setItem(i, 2, QTableWidgetItem(issue["desc"]))
            
        self.table.itemDoubleClicked.connect(self.on_double_click)
        layout.addWidget(self.table)
        
        # 🌟 [추가] 하단 버튼 레이아웃 분리 및 다중 열기 버튼 추가
        btn_layout = QHBoxLayout()
        
        btn_fix_all = QPushButton("🛠️ 문제 있는 이미지만 모아서 라벨링 툴 열기")
        btn_fix_all.setStyleSheet("background-color: #fef08a; font-weight: bold; color: #854d0e;")
        btn_fix_all.setMinimumHeight(40)
        btn_fix_all.clicked.connect(self.on_fix_all_clicked)
        
        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.accept)
        btn_close.setMinimumHeight(40)
        
        btn_layout.addWidget(btn_fix_all)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

    def on_double_click(self, item):
        row = item.row()
        file_name = self.table.item(row, 0).text()
        self.request_fix.emit(file_name)
        self.accept()

    # 🌟 [추가] 다중 열기 버튼 클릭 이벤트
    def on_fix_all_clicked(self):
        file_names = list(set([issue["file"] for issue in self.issues if "file" in issue]))
        self.request_fix_multi.emit(file_names)
        self.accept()
