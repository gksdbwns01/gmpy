from .utils import *
from .utils.discord import send_discord_webhook
from .utils.logger import logger
from .ui.main_window import MainWindow


def main():
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    app = QApplication(sys.argv)
    font_name = "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"
    app.setFont(QFont(font_name, 10))
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    
    window = MainWindow()
    
    # UI main-process fallback for uncaught exceptions.
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        if hasattr(window, 'webhook_url') and window.webhook_url:
            error_details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            if len(error_details) > 1000:
                error_details = error_details[-1000:] 
                
            send_discord_webhook(
                webhook_url=window.webhook_url,
                title="?? [???? ??? ??]",
                description=f"UI ?? ?????? ???? ?? ??? ???? ?????.\n```{error_details}```",
                color=0x992d22,
                sync=True
            )
            
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = global_exception_handler
    
    window.showMaximized()
    return app.exec_()
