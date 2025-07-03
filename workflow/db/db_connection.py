import sqlite3

DB_PATH = "../data/dataset/fund_data.db"

def query_db(sql: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql)
    result = cur.fetchall()
    conn.close()
    return result
