#!/bin/bash

# 清除屏幕
clear

python -m workflow.text2sql_dataset_generator.run

python -m workflow.text2sql_dataset_generator.merge_datasets