#!/bin/bash

# 清除屏幕
clear

python -m workflow.train_text_sql.deepspeed_trainer.scripts.server --trust_remote_code\
    --host 127.0.0.1 --port 8899