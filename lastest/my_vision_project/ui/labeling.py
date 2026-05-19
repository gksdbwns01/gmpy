from ..utils import *
from ..core import *

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
        
        img_array = np.fromfile(str(self.current_image_path), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: return False, "이미지를 읽을 수 없습니다."
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
        
        import cv2, numpy as np; 
        img_array = np.fromfile(str(self.current_image_path), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None: return False, "이미지를 읽을 수 없습니다."
        self.save_state(); total_added = 0
        for tmpl_data in self.saved_templates:
            template, class_id, w, h = tmpl_data['template'], tmpl_data['class_id'], tmpl_data['w'], tmpl_data['h']
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
