#!/bin/bash

# 清除屏幕
clear
  
deepspeed --module workflow.train_text_sql.deepspeed_trainer.scripts.train

