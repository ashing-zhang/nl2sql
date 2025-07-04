#!/bin/bash

# 清除屏幕
clear

# 参数定义
num_rounds=20
save_data_path="workflow/text2sql_dataset_generator/test_text_sql.json"
db_url="data/dataset/fund_data.db"

python -m workflow.text2sql_dataset_generator.run --num_rounds $num_rounds --save_data_path $save_data_path\
    --db_url $db_url

python -m workflow.text2sql_dataset_generator.merge_datasets