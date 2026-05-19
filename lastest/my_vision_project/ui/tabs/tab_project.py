from ...utils import *
from ...core import *
from ...core.workers import *
from ..ui_components import *


class ProjectTabMixin:
    def export_project_dialog(self):
        logger.info("프로젝트 압축(내보내기) 실행")
        t3_model = self.t3_model.get_path() if hasattr(self, 't3_model') else ""
        t4_model = self.t4_eval_model_display.text() if hasattr(self, 't4_eval_model_display') else ""
        has_base = bool(t3_model and Path(t3_model).exists())
        has_retrained = bool(t4_model and Path(t4_model).exists())
        if not has_base and not has_retrained: QMessageBox.warning(self, "경고", "내보낼 모델이 없습니다."); return
        proj_name = Path(self.w_proj_root.get_path()).name; save_path, _ = QFileDialog.getSaveFileName(self, "프로젝트 내보내기 (Zip)", f"{proj_name}.zip", "Zip Files (*.zip)")
        if not save_path: return
        config_data = self.config_builder.build(self)
        if "global" in config_data and "webhook_url" in config_data["global"]: config_data["global"]["webhook_url"] = ""
        final_json_data = {"metadata": {"saved_at": datetime.now().strftime("%Y%m%d_%H%M%S"), "type": "project_snapshot", "has_base_model": has_base, "has_retrained_model": has_retrained}, "config": config_data}
        self.statusBar().showMessage("📦 프로젝트 내보내기 중...")
        try:
            with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("config.json", json.dumps(final_json_data, ensure_ascii=False, indent=4))
                if has_base: zf.write(t3_model, "base_model.pt")
                if has_retrained: zf.write(t4_model, "retrained_model.pt")
            logger.info(f"프로젝트 내보내기 완료: {save_path}")
            QMessageBox.information(self, "내보내기 완료", f"저장되었습니다:\n{save_path}"); self.statusBar().showMessage(f"✅ 프로젝트 내보내기 완료: {Path(save_path).name}", 5000)
        except Exception as e: 
            logger.error(f"프로젝트 내보내기 중 에러: {e}", exc_info=True)
            QMessageBox.critical(self, "오류", f"오류 발생:\n{e}"); self.statusBar().clearMessage()
    def import_project_dialog(self):
        logger.info("프로젝트 복구(불러오기) 실행")
        load_path, _ = QFileDialog.getOpenFileName(self, "프로젝트 불러오기 (Zip)", "", "Zip Files (*.zip)")
        if not load_path: return
        import_name = Path(load_path).stem; current_root_parent = Path(self.w_proj_root.get_path()).parent; new_proj_path = current_root_parent / import_name; is_renamed = False; original_name = new_proj_path.name; counter = 1
        while new_proj_path.exists(): is_renamed = True; new_proj_path = current_root_parent / f"{import_name}_{counter}"; counter += 1
        new_proj_path.mkdir(parents=True, exist_ok=True); target_dir = new_proj_path / "workspace" / f"Imported_{import_name}"; target_dir.mkdir(parents=True, exist_ok=True)
        self.statusBar().showMessage(f"📥 '{new_proj_path.name}' 프로젝트를 구성하는 중..."); QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            with zipfile.ZipFile(load_path, 'r') as zf:
                resolved_target = Path(target_dir).resolve()
                for member in zf.infolist():
                    member_path = Path(target_dir / member.filename).resolve()
                    if not str(member_path).startswith(str(resolved_target)):
                        raise PermissionError(f"보안 경고: 압축 파일이 지정된 경로를 벗어나려고 합니다! ({member.filename})")
                zf.extractall(target_dir)

            config_file = target_dir / "config.json"; config_data = {}
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f: config_data = json.load(f).get("config", {})
            self.w_proj_root.line_edit.setText(str(new_proj_path)); self.apply_loaded_config(config_data)
            legacy_model, base_model, retrained_model = target_dir / "model.pt", target_dir / "base_model.pt", target_dir / "retrained_model.pt"
            if base_model.exists() or legacy_model.exists():
                model_path_str = str((base_model if base_model.exists() else legacy_model).resolve()); self.t3_model.line_edit.setText(model_path_str); self.t4_base.line_edit.setText(model_path_str); self.t5_model.line_edit.setText(model_path_str)
            if retrained_model.exists(): retrained_path_str = str(retrained_model.resolve()); self.t4_eval_model_display.setText(retrained_path_str); self.t4_btn_eval.setEnabled(True); self.t4_btn_auto_thr.setEnabled(True); self.t5_model.line_edit.setText(retrained_path_str)
            self.config_manager.update_workspace_path(str(new_proj_path / "workspace"))
            QApplication.restoreOverrideCursor()
            logger.info(f"프로젝트 불러오기 완료: {new_proj_path}")
            if is_renamed: 
                QMessageBox.warning(self, "⚠️ 이름 변경", f"기존에 '{original_name}' 폴더가 존재하여 이름이 변경되었습니다!\n새 폴더명: {new_proj_path.name}"); self.statusBar().showMessage(f"⚠️ 폴더 이름 변경됨: {new_proj_path.name} 로 복구 완료", 7000)
            else: 
                QMessageBox.information(self, "✅ 불러오기 완료", f"루트: {new_proj_path.name}\n모든 경로가 연동되었습니다."); self.statusBar().showMessage(f"✅ 프로젝트 불러오기 완료: {new_proj_path.name}", 5000)
        except Exception as e: 
            logger.error(f"프로젝트 불러오기 중 에러: {e}", exc_info=True)
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "❌ 오류", f"오류 발생:\n{e}"); self.statusBar().clearMessage()
