'''
    python -m workflow.text2sql_dataset_generator.run
'''
from .database.connector import DatabaseConnector
from .generator.base_query_generator import QueryGenerator
# from .generator.add_query_generator import QueryGenerator
from .run_sql_from_json import execute_sqls_from_json
import json

def main():
    schema_path = "workflow/text2sql_dataset_generator/schema.json"
    db_url = "data/dataset/fund_data.db"
    save_data_path = "workflow/text2sql_dataset_generator/test_text_sql.json"
    # 初始化模块
    db = DatabaseConnector(schema_path,db_url)
    db.save_schema_json()
    
    generator = QueryGenerator(schema_path)
    db_schema = db.extract_schema()
    
    dataset = []
    identity_id = 0  # 唯一标识计数器

    for i in range(200):
        queries = generator.generate_queries(db_schema)
        print(f'queries {i}:', queries)
        
        for q in queries:
            # 重构数据结构
            formatted_data = {
                "conversations": [
                    {
                        "from": "user",
                        "value": q["question"].strip()
                    },
                    {
                        "from": "assistant",
                        "value": q["sql"].strip() 
                    }
                ]
            }
            dataset.append(formatted_data)
            identity_id += 1  # 标识自增
    
    with open(save_data_path, 'w', encoding='utf-8') as f:  
        json.dump(
            dataset, 
            f, 
            indent=2,
            ensure_ascii=False  # 禁用ASCII转义
        )
    # 数据清洗
    execute_sqls_from_json(save_data_path,db_url)

if __name__ == "__main__":
    main()