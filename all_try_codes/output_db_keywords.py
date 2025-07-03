import json

'''
    将fund_data.db数据库中的所有关键词（表名、字段名等）都输出到data/dataset/db_keywords.txt文件中
'''
def output_db_keywords(db_schema_path):
    with open(db_schema_path, "r") as f:
        db_schema = json.load(f)
    
    db_keywords = set()
    # Extract keywords from the database schema
    for table, columns in db_schema.items():
        db_keywords.add(table)
        for column in columns:
            db_keywords.add(column['name'])
    
    with open("data/dataset/db_keywords.txt", "w") as f:
        for keyword in db_keywords:
            f.write(keyword + "\n")

if __name__ == '__main__':
    db_schema_path = 'data/dataset/db_schema.json'
    output_db_keywords(db_schema_path)
