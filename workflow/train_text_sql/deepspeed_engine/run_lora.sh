#!/bin/bash

# 清除屏幕
clear

DS_CONFIG_PATH="ds_config.json"

deepspeed finetune.py --deepspeed --deepspeed_config $DS_CONFIG_PATH 
  
