#!/bin/bash

# 清除屏幕
clear

DS_CONFIG_PATH="workflow/train_text_sql/deepspeed_engine/ds_config.json"

# 调试模式  
deepspeed workflow/train_text_sql/deepspeed_engine/finetune.py --deepspeed --deepspeed_config $DS_CONFIG_PATH --debug
# 训练模式
# deepspeed workflow/train_text_sql/deepspeed_engine/finetune.py --deepspeed --deepspeed_config $DS_CONFIG_PATH
