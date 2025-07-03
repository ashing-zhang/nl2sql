import sqlite3
import json
import re

class DatabaseConnector:
    def __init__(self, schema_path,db_url):
        # 移除SQLAlchemy相关初始化
        self.conn = None
        self.cursor = None
        self.schema_path = schema_path
        
        # 建立原生SQLite连接[6,12](@ref)
        try:
            self.conn = sqlite3.connect(db_url)
            self.conn.execute("PRAGMA foreign_keys = ON")  # 启用外键约束[10](@ref)
            self.cursor = self.conn.cursor()
            print("数据库连接成功")
        except sqlite3.Error as e:
            print(f"连接失败: {str(e)}")

    def extract_schema(self):
        schema = {}
        try:
            # 获取所有表名
            # sqlite_master 是 SQLite 的内置表，存储数据库的元数据。
            # type='table' 筛选出所有用户创建的表（排除视图、索引等）
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            #  将查询结果转换为表名列表，存储在 tables 中
            tables = [row[0] for row in self.cursor.fetchall()]
            # print("tables:",tables)

            for table in tables:
                # 提取列基础信息
                # PRAGMA table_info 是 SQLite 的命令，用于获取表的列信息
                self.cursor.execute(f"PRAGMA table_info({table})")
                '''
                    cid（row[0]）：列的索引。
                    name（row[1]）：列名。
                    type（row[2]）：数据类型（如 INTEGER, TEXT）。
                    notnull（row[3]）：是否允许为空（0 或 1）。
                    default_value（row[4]）：默认值。
                    pk（row[5]）：是否为主键（0 表示否，1 或其他非零值表示是）。
                '''
                columns = {
                    row[1]: {
                        "type": row[2].upper(),
                        "pk": bool(row[5]),
                    } for row in self.cursor.fetchall()
                }

                # 合并外键信息
                self.cursor.execute(f"PRAGMA foreign_key_list({table})")
                foreign_keys = {}  
                for fk in self.cursor.fetchall():
                    # fk数据结构：(id, seq, table, from_col, to_col)
                    child_col = fk[3]  # 子表列名
                    parent_table = fk[2]      # 父表名称
                    parent_col = fk[4]        # 父表列名
                    
                    foreign_keys[child_col] = {
                        "ref_table": parent_table,
                        "ref_column": parent_col  # 直接使用父列名而非再次查询
                    }
                schema[table] = {
                    "columns": columns,
                    "foreign_keys": foreign_keys
                }
            return schema
        except sqlite3.Error as e:
            print(f"元数据提取失败: {str(e)}")
            return {}

    def save_schema_json(self):
        """保存表结构到JSON文件"""
        with open(self.schema_path, 'w', encoding='utf-8') as f:
            json.dump(self.extract_schema(), f, indent=2, ensure_ascii=False)

    def __del__(self):
        if self.conn:
            self.conn.close()