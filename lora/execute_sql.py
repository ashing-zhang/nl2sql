'''
    批量执行人工构造nl-sql训练数据中的sql，确保所造的sql语句正确且顺利执行
    cd to lora dir and python execute_sql.py
'''
import os
import sqlite3

def convert_txt_to_sql(folder_path):
    """将文件夹内所有txt文件转换为同名sql文件"""
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            txt_path = os.path.join(folder_path, filename)
            sql_filename = filename.replace('.txt', '.sql')
            sql_path = os.path.join(folder_path, sql_filename)
            
            with open(txt_path, 'r', encoding='utf-8') as f_txt:
                sql_content = f_txt.read()
            
            with open(sql_path, 'w', encoding='utf-8') as f_sql:
                f_sql.write(sql_content)
            
            print(f"生成SQL文件：{sql_filename}")

import time
import threading

# 全局线程锁防止并发写入冲突
db_lock = threading.Lock()

def execute_sql_files(db_path, folder_path, log_path='./data/execute_log.txt'):
    """执行SQL文件并优化数据库锁定处理"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    try:
        # 设置30秒超时和WAL日志模式
        with db_lock:  # 确保单线程访问
            conn = sqlite3.connect(db_path, timeout=30)
            conn.execute('PRAGMA journal_mode=WAL;')
            
            with conn:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n{'='*50}\n执行日志 {os.path.basename(db_path)}\n")
                    
                    for sql_file in sorted(f for f in os.listdir(folder_path) if f.endswith('.sql')):
                        full_path = os.path.join(folder_path, sql_file)
                        print(f"正在处理：{sql_file}")
                        
                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                sql_content = f.read().strip()
                            
                            if not sql_content:
                                log_file.write(f"\n\n=== 空文件跳过：{sql_file} ===")
                                continue
                                
                            log_entry = f"\n\n=== 执行文件：{sql_file} ===\n"
                            max_retries = 3  # 最大重试次数[5,8](@ref)
                            
                            for statement in sql_content.split(';'):
                                statement = statement.strip()
                                if not statement:
                                    continue
                                
                                for attempt in range(max_retries):
                                    try:
                                        cursor = conn.cursor()
                                        cursor.execute(statement)
                                        
                                        # 处理查询结果
                                        if cursor.description:
                                            results = cursor.fetchmany(5)
                                            log_entry += f"\n[查询结果-前5条]:\n" + '\n'.join(str(r) for r in results)
                                        
                                        conn.commit()  # 及时提交事务
                                        break
                                        
                                    except sqlite3.OperationalError as e:
                                        if 'locked' in str(e) and attempt < max_retries-1:
                                            wait_time = 2 ** attempt  # 指数退避[5](@ref)
                                            time.sleep(wait_time)
                                            continue
                                        raise
                                        
                            log_file.write(log_entry)
                            print(f"完成处理：{sql_file}")
                            
                        except Exception as e:
                            error_msg = f"\n\n=== 文件处理失败：{sql_file} ===\n错误类型：{type(e).__name__}\n详细信息：{str(e)}"
                            log_file.write(error_msg)
                            print(error_msg)
                            
    except Exception as e:
        error_msg = f"\n\n!!! 严重错误 !!!\n数据库连接失败：{str(e)}"
        with open(log_path, 'a') as f:
            f.write(error_msg)
        print(error_msg)
        
    finally:
        if 'conn' in locals():
            conn.close()

def delete_sql_recursive(folder_path):
    """递归删除所有子目录中的.sql文件"""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.sql'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"已删除：{file_path}")
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")

if __name__ == "__main__":
    folder_path = 'data/train_label'
    db_path = '../data/dataset/fund_data.db'
    log_path = './data/execute_log.txt'

    # 日志文件清空操作
    if os.path.exists(log_path):
        with open(log_path, 'w', encoding='utf-8') as f:  # 'w'模式自动清空文件内容
            pass
        print(f"已清空日志文件：{log_path}")
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # delete_sql_recursive(folder_path)
    # 分步执行
    convert_txt_to_sql(folder_path)
    # execute_sql_files(db_path, folder_path, log_path)