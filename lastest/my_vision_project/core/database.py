from ..utils import *
from ..utils.logger import logger

class LogDatabase:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        logger.debug(f"LogDatabase 초기화. 경로: {self.db_path}")
        self._ensure_db()
        self.log_queue = qlib.Queue()
        self.writer_thread = threading.Thread(target=self._db_writer_loop, daemon=True)
        self.writer_thread.start()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=30000000000")
        return conn

    def _ensure_db(self):
        if not self.db_path.parent.exists(): 
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS training_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, project_path TEXT, task_type TEXT, model_name TEXT, epochs INTEGER, batch_size INTEGER, best_map REAL, save_dir TEXT, config_json TEXT)')
            cursor.execute('CREATE TABLE IF NOT EXISTS evaluation_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, project_path TEXT, task_type TEXT, model_name TEXT, total_imgs INTEGER, wrong_count INTEGER, accuracy REAL, wrong_images TEXT, config_json TEXT)')
            conn.commit()

    def _db_writer_loop(self):
        conn = self._get_connection()
        while True:
            task = self.log_queue.get()
            if task is None: break
            try:
                task(conn)
            except Exception as e:
                logger.error(f"DB 쓰기 에러 발생: {e}", exc_info=True)

    def insert_log(self, project_path, task_type, model_name, epochs, batch, best_map, save_dir, config_data):
        params = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            project_path,
            task_type, 
            model_name, 
            epochs, 
            batch, 
            best_map, 
            save_dir, 
            json.dumps(config_data, ensure_ascii=False)
        )
        def _task(conn):
            conn.execute('''
                INSERT INTO training_logs 
                (timestamp, project_path, task_type, model_name, epochs, batch_size, best_map, save_dir, config_json) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', params)
            conn.commit()
            from pathlib import Path
            logger.info(f"학습 로그 DB 삽입 완료. Project: {Path(project_path).name}, Task: {task_type}, Best mAP: {best_map:.4f}")
            
        self.log_queue.put(_task)

    def insert_eval_log(self, project_path, task_type, model_name, total, wrong, accuracy, wrong_imgs_list, config_data):
        params = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            project_path,
            task_type, 
            model_name, 
            total, 
            wrong, 
            accuracy, 
            json.dumps([Path(p).name for p in wrong_imgs_list], ensure_ascii=False), 
            json.dumps(config_data, ensure_ascii=False)
        )
        def _task(conn):
            conn.execute('''
                INSERT INTO evaluation_logs 
                (timestamp, project_path, task_type, model_name, total_imgs, wrong_count, accuracy, wrong_images, config_json) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', params)
            conn.commit()
            from pathlib import Path
            logger.info(f"평가 로그 DB 삽입 완료. Project: {Path(project_path).name}, Task: {task_type}, Acc: {accuracy:.2f}%")
        self.log_queue.put(_task)

    def delete_logs(self, table_type, ids):
        if not ids: return
        table_name = 'training_logs' if table_type == 'train' else 'evaluation_logs'
        placeholders = ','.join('?' for _ in ids)
        evt = threading.Event()
        def _task(conn):
            conn.execute(f'DELETE FROM {table_name} WHERE id IN ({placeholders})', ids)
            conn.commit()
            evt.set()
        self.log_queue.put(_task)
        evt.wait() # UI 쓰레드가 삭제 완료를 대기함
        logger.info(f"DB 레코드 삭제 완료. Table: {table_name}, IDs: {ids}")

    def fetch_logs(self, table_type, current_proj_path="", search_kw="", date_from=None, date_to=None, filter_ng=False, offset=0, limit=100, sort_col="id", sort_order="DESC"):
        table_name = 'training_logs' if table_type == 'train' else 'evaluation_logs'
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        params = []
        if current_proj_path:
            query += " AND project_path = ?"
            params.append(current_proj_path)
        if search_kw:
            query += " AND (task_type LIKE ? OR model_name LIKE ?)"
            params.extend([f"%{search_kw}%", f"%{search_kw}%"])
        if date_from and date_to:
            query += " AND timestamp BETWEEN ? AND ?"
            params.extend([date_from + " 00:00:00", date_to + " 23:59:59"])
        if table_type == 'eval' and filter_ng:
            query += " AND wrong_count > 0"
            
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        try:
            with self._get_connection() as conn:
                total = conn.execute(count_query, params).fetchone()[0]
                if limit > 0:
                    query += f" ORDER BY {sort_col} {sort_order} LIMIT ? OFFSET ?"
                    params.extend([limit, offset])
                else:
                    query += f" ORDER BY {sort_col} {sort_order}"
                rows = conn.execute(query, params).fetchall()
            logger.debug(f"DB 로그 조회 완료. Table: {table_name}, Count: {len(rows)}/{total}")
            return rows, total
        except Exception as e:
            logger.error(f"DB 로그 조회 중 오류: {e}", exc_info=True)
            return [], 0
