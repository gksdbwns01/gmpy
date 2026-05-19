from .common import *
from .logger import logger

def kill_process_tree(pid):
    """지정된 PID와 그 자식 프로세스들까지 OS 레벨에서 확실하게 사살합니다"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
        logger.info(f"프로세스 트리(PID: {pid}) 및 하위 프로세스 강제 종료 완료.")
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.error(f"프로세스 트리 강제 종료 중 예외 발생: {e}")

def clear_vram():
    logger.debug("VRAM 정리(Garbage Collection & Cuda empty cache) 시작")
    gc.collect()
    try:
        if torch.cuda.is_available(): torch.cuda.empty_cache(); logger.debug("CUDA empty_cache 완료")
    except Exception as e: 
        logger.error(f"VRAM 정리 중 오류 발생: {e}", exc_info=True)


class GracefulStopHandler:
    def __init__(self, stop_event, logger, result_dict=None):
        self.stop_event = stop_event
        self.logger = logger
        self.result_dict = result_dict

    def should_stop(self, context=""):
        if self.stop_event and self.stop_event.is_set():
            self.logger.info(f"🛑 사용자 중단 신호 감지 [{context}]")
            if self.result_dict is not None:
                self.result_dict["error"] = f"작업이 취소되었습니다. ({context})"
            return True
        return False

    def check_every_n_iterations(self, iteration, interval=10, context=""):
        if iteration % interval == 0:
            return self.should_stop(context)
        return False

def open_folder(path):
    p = Path(path)
    if not p.exists(): logger.warning(f"폴더를 열 수 없습니다. 존재하지 않음: {path}"); return
    target_dir = str(p.parent) if p.is_file() else str(p)
    logger.debug(f"폴더 열기 실행: {target_dir}")
    if platform.system() == "Windows": os.startfile(target_dir)
    elif platform.system() == "Darwin": subprocess.Popen(["open", target_dir])
    else: subprocess.Popen(["xdg-open", target_dir])
