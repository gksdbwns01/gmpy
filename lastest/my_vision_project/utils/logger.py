from .common import *

class EmojiFormatter(logging.Formatter):
    LEVEL_EMOJIS = {
        logging.DEBUG: "🔍 [DEBUG]",
        logging.INFO: "ℹ️ [INFO]",
        logging.WARNING: "⚠️ [WARN]",
        logging.ERROR: "❌ [ERROR]",
        logging.CRITICAL: "🔴 [CRITICAL]"
    }

    def format(self, record):
        original_levelname = record.levelname
        record.levelname = self.LEVEL_EMOJIS.get(record.levelno, original_levelname)
        result = super().format(record)
        record.levelname = original_levelname
        return result

def setup_logger():
    logger = logging.getLogger("YOLO_Pipeline")
    if logger.hasHandlers(): return logger
    logger.setLevel(logging.DEBUG)
    
    base_dir = Path(sys.executable).parent if getattr(sys, 'frozen', False) else get_app_base_dir()
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    file_handler = RotatingFileHandler(log_dir / "app.log", maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('[%(levelname)s] %(asctime)s - [PID:%(process)d|%(threadName)s] - %(funcName)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = EmojiFormatter('%(levelname)s %(asctime)s - [%(threadName)s] - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger

logger = setup_logger()
