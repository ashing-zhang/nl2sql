import sqlite3
import json
from pathlib import Path

class DatabaseExporter:
    def __init__(self, db_path='./data/dataset/fund_data.db'):
        self.db_path = db_path
        self._verify_db_connection()

    def _verify_db_connection(self):
        """验证数据库连接是否正常"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.close()
        except Exception as e:
            raise ConnectionError(f"数据库连接失败: {str(e)}")

    def list_tables(self):
        """显示数据库中包含的所有表的表名"""
        print(f"正在检查数据库路径: {self.db_path}")  # 添加路径确认
        if not Path(self.db_path).is_file():
            print(f"数据库路径无效: {self.db_path}")
            return None
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if not tables:
                    print("数据库中没有表")
                    return []
                table_names = [table[0] for table in tables]
                print("所有表名:", table_names)
                return table_names
        except sqlite3.Error as e:
            print(f"数据库操作失败: {str(e)}")
            return None
        except Exception as e:
            print(f"未知错误: {str(e)}")
            return None

    def _sanitize_column_names(self):
        """将数据库中所有带括号的字段名替换为下划线，并输出所有表的所有字段到文件"""
        all_columns = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                tables = self.list_tables()
                for table in tables:
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    for col in columns:
                        old_name = col[1]
                        new_name = old_name.replace('(', '_').replace(')', '_')
                        if old_name != new_name:
                            cursor.execute(f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}")
                        all_columns.append(new_name)
            with open('all_columns.txt', 'w', encoding='utf-8') as f:
                for column in all_columns:
                    f.write(f"{column}\n")
            print("所有字段已输出到文件: all_columns.txt")
        except Exception as e:
            print(f"操作失败: {str(e)}")

    def export_schema_to_json(self, output_file='db_schema.json'):
        """将数据库中所有表的表名和字段信息导出到JSON文件"""
        schema = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                tables = self.list_tables()
                for table in tables:
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    schema[table] = [{"name": col[1], "type": col[2]} for col in columns]
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(schema, f, ensure_ascii=False, indent=4)
            print(f"数据库模式已导出到文件: {output_file}")
        except Exception as e:
            print(f"操作失败: {str(e)}")

# 使用示例
if __name__ == "__main__":
    db_path='./data/dataset/fund_data.db'
    exporter = DatabaseExporter(db_path)

    # 调用方法修改字段名并输出所有字段
    # exporter._sanitize_column_names()

    # 调用方法导出数据库模式到JSON文件
    exporter.export_schema_to_json()
