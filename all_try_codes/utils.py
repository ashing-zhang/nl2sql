'''
    工具函数
'''
from huggingface_hub import snapshot_download
import json
import sqlite3

def download_embedding_model(local_model_path,repo_id):
    # 手动下载
    snapshot_download(repo_id=repo_id, local_dir=local_model_path)

def parse_database_schema(db_path, db_schema_path):
        # 初始化数据库连接
        conn = sqlite3.connect(db_path)
        schema = {}
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        # 获取数据库模式中的所有列名
        all_columns = set()
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
            all_columns.update([col["name"] for col in columns])
            schema[table] = columns
        
        with open(db_schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=4)
        conn.close()


if __name__ == '__main__':
    db_path = "data/dataset/fund_data.db"
    db_schema_path = "data/dataset/db_schema.json"
    parse_database_schema(db_path, db_schema_path)