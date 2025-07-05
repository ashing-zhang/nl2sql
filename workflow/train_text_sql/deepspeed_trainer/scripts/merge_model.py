#!/usr/bin/env python3
"""
Merge base model and LoRA adapter weights into a single model for inference.
Usage:
    python -m workflow.train_text_sql.deepspeed_trainer.scripts.merge_model\
         --base_model BASE_MODEL_PATH --adapter_path LORA_ADAPTER_PATH --merged_model MERGED_MODEL_OUTPUT_DIR
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Merge base model and LoRA adapter weights.")
    parser.add_argument('--base_model', type=str, required=True, help='Path to the base model directory')
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to the LoRA adapter directory')
    parser.add_argument('--merged_model', type=str, required=True, help='Directory to save the merged model')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading base model from: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    print(f"Loading LoRA adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    print(f"Saving merged model to: {args.merged_model}")
    merged_model.save_pretrained(args.merged_model)
    # 保存分词器（可选）
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.merged_model)
    print("Merge complete!")

if __name__ == "__main__":
    main() 