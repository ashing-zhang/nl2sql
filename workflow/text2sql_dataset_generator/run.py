'''
    python -m workflow.text2sql_dataset_generator.run
'''
from .database.connector import DatabaseConnector
from .generator.base_query_generator import QueryGenerator
# from .generator.add_query_generator import QueryGenerator
from .run_sql_from_json import execute_sqls_from_json
import json
import argparse

def main(num_rounds, save_data_path, db_url):
    schema_path = "workflow/text2sql_dataset_generator/schema.json"
    # 初始化模块
    db = DatabaseConnector(schema_path,db_url)
    db.save_schema_json()
    
    generator = QueryGenerator(schema_path)
    db_schema = db.extract_schema()
    
    dataset = []
    identity_id = 0  # 唯一标识计数器

    for i in range(num_rounds):
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
    parser = argparse.ArgumentParser(description="Generate text2sql dataset.")
    parser.add_argument('--num_rounds', type=int, default=200, help='生成轮数（每轮生成若干条问答对）')
    parser.add_argument('--save_data_path', type=str, default="workflow/text2sql_dataset_generator/test_text_sql.json", help='保存生成数据的路径')
    parser.add_argument('--db_url', type=str, default="data/dataset/fund_data.db", help='sqlite数据库地址')
    args = parser.parse_args()
    main(num_rounds=args.num_rounds, save_data_path=args.save_data_path, db_url=args.db_url)