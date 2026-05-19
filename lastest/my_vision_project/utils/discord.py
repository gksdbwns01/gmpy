from .common import *
from .logger import logger

def send_discord_webhook(webhook_url, title, description, color=0x3498db, fields=None, retry_count=3, sync=False):
    if not webhook_url or not webhook_url.startswith("http"): 
        return False

    def _send_task():
        import requests
        import time
        from datetime import datetime
        from requests.exceptions import Timeout, ConnectionError, RequestException

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        if fields:
            embed["fields"] = fields

        payload = {"embeds": [embed]}

        for attempt in range(retry_count):
            try:
                logger.debug(f"Discord 임베드 웹훅 전송 시도. (시도 {attempt+1}/{retry_count})")
                response = requests.post(
                    webhook_url, 
                    json=payload, 
                    timeout=5
                )
                
                # 디스코드 Rate Limit (429) 처리
                if response.status_code == 429:
                    try:
                        retry_after = response.json().get("retry_after", 2.0)
                    except:
                        retry_after = 2.0
                    logger.warning(f"웹훅 Rate Limit 도달. {retry_after}초 대기 후 재시도...")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                logger.debug("웹훅 전송 성공")
                return
                
            except Timeout:
                logger.warning(f"웹훅 타임아웃 (시도 {attempt+1}/{retry_count})")
            except ConnectionError as e:
                logger.error(f"네트워크 연결 실패: {e}")
            except RequestException as e:
                logger.error(f"웹훅 HTTP 에러: {e}")
            except Exception as e:
                logger.error(f"웹훅 전송 중 알 수 없는 오류: {e}")
            
            if attempt < retry_count - 1:
                time.sleep(2)

        logger.error("디스코드 웹훅 최종 전송 실패")
        
    if sync:
        _send_task()  # 스레드를 쓰지 않고 그 자리에서 전송을 끝낼 때까지 대기
    else:
        import threading
        threading.Thread(target=_send_task, daemon=True).start()  # 평소처럼 백그라운드 전송
    return True

def create_heartbeat_callback(webhook_url, total_epochs, interval):
    if interval <= 0: return lambda trainer: None
    
    def on_train_epoch_end(trainer):
        current_epoch = trainer.epoch + 1
        if current_epoch % interval == 0:
            logger.debug(f"웹훅 Heartbeat 발생: Epoch {current_epoch}/{total_epochs}")
            
            # tloss 계산 중 발생할 수 있는 Ultralytics 내부 에러 방어
            loss_val = "N/A"
            try:
                if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                    loss_val = f"`{trainer.tloss.sum().item():.4f}`"
            except Exception as e:
                logger.warning(f"웹훅: Loss 값 추출 실패 ({e})")

            fields = [
                {"name": "진척도 (Epochs)", "value": f"{current_epoch} / {total_epochs}", "inline": True},
                {"name": "Total Loss", "value": loss_val, "inline": True}
            ]
            
            send_discord_webhook(
                webhook_url=webhook_url,
                title="💓 [학습 진행 상황]",
                description="모델 학습이 정상적으로 진행 중입니다.",
                color=0x3498db,
                fields=fields,
                sync=True # 워커 프로세스 종료 전 누락 방지를 위한 동기화 옵션 추가
            )
    return on_train_epoch_end

def create_stop_callback(stop_event):
    def check_stop(trainer):
        if stop_event is not None and stop_event.is_set():
            logger.info("🛑 사용자의 중지 요청 감지. 작업을 안전하게 조기 종료합니다.")
            trainer.stop = True
    return {
        "on_train_epoch_end": check_stop,
        "on_train_batch_end": check_stop
    }
