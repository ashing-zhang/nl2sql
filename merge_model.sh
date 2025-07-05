#!/bin/bash

# 清除屏幕
clear

BASE_MODEL_PATH=workflow/models/Qwen2.5-7B-Instruct
LORA_ADAPTER_PATH=workflow/train_text_sql/deepspeed_trainer/outputs
MERGED_MODEL_OUTPUT_DIR=workflow/train_text_sql/deepspeed_trainer/merged_model

python -m workflow.train_text_sql.deepspeed_trainer.scripts.merge_model \
    --base_model "$BASE_MODEL_PATH" \
    --adapter_path "$LORA_ADAPTER_PATH" \
    --merged_model "$MERGED_MODEL_OUTPUT_DIR"