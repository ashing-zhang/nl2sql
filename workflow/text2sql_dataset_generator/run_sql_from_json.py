'''
    数据清洗：将执行效率低下或执行失败的SQL语句从数据集中删除
'''
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

class SQLiteQueryController:
    def __init__(self, db_path):
        self.db_path = db_path
        '''
            当锁处于 locked 状态时，其他尝试获取锁的线程将被阻塞 (block)，
            直到持有锁的线程释放 (release) 它
        '''
        self.lock = threading.Lock()
        self.active_connections = {}  # {thread_id: connection}
        self.query_timeout = 120  # 单位：秒

    def _create_connection(self):
        """创建线程专用连接"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=120,
            check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL")  # 启用WAL模式提升并发
        conn.execute("PRAGMA busy_timeout=120000")  # 设置120秒锁等待
        return conn

    def execute_with_timeout(self, sql):
        """带超时控制的查询执行"""
        thread_id = threading.get_ident()
        result = {"status": "pending"}
        
        def _worker():
            try:
                # 自动获取锁，并在块结束时（无论正常或异常）自动释放
                with self.lock:
                    if thread_id not in self.active_connections:
                        self.active_connections[thread_id] = self._create_connection()
                    conn = self.active_connections[thread_id]
                
                cursor = conn.cursor()
                cursor.execute(sql)
                cursor.fetchall()  # 确保完全获取结果
                result["status"] = "success"
            except sqlite3.OperationalError as e:
                result.update({"status": "error", "message": str(e)})
            except Exception as e:
                result.update({"status": "fatal", "message": str(e)})
            finally:
                if conn: conn.commit()

        # 启动监控线程
        def _timeout_monitor():
            worker_thread = threading.Thread(target=_worker)
            worker_thread.start()
            worker_thread.join(self.query_timeout)
            if worker_thread.is_alive():
                with self.lock:
                    if conn := self.active_connections.get(thread_id):
                        conn.interrupt()  # 强制终止查询
                result.update({"status": "timeout"})

        monitor_thread = threading.Thread(target=_timeout_monitor)
        monitor_thread.start()
        monitor_thread.join()
        return result

def execute_sqls_from_json(data_path, db_path):
    controller = SQLiteQueryController(db_path)
    valid_data = []
    removed_count = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def process_item(item):
        nonlocal removed_count
        try:
            for msg in item["conversations"]:
                if msg["from"] == "assistant":
                    sql = msg["value"]
                    result = controller.execute_with_timeout(sql)
                    if result["status"] != "success":
                        print(f"❌ 执行失败 [{result['status']}]: {sql[:50]}...")
                        return False
            print(f"✅ 执行成功: {item['id']}")
            return True
        except Exception as e:
            print(f"⚠️ 未知异常: {str(e)[:50]}...")
            return False

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_item, item): item for item in data}
        
        for future in futures:
            item = futures[future]
            try:
                if future.result(timeout=controller.query_timeout + 5):
                    valid_data.append(item)
                else:
                    removed_count += 1
            except TimeoutError:
                print(f"⏰ 全局超时终止: {item['id']}")
                removed_count += 1

    # 写回有效数据
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 清洗完成 - 保留{len(valid_data)}条，删除{removed_count}条")

if __name__ == "__main__":
    # execute_sqls_from_json("text2sql_dataset.json", "../../data/dataset/fund_data.db")
    execute_sqls_from_json("text2sql_dataset_add.json", "../../data/dataset/fund_data.db")