#!/bin/bash

# 清除屏幕
clear

# 参数定义
num_rounds=10
save_data_path="workflow/text2sql_dataset_generator/test_text_sql.json"

python -m workflow.text2sql_dataset_generator.run --num_rounds $num_rounds --save_data_path $save_data_path

python -m workflow.text2sql_dataset_generator.merge_datasets