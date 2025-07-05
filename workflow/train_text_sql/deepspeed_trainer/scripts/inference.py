#!/usr/bin/env python3
"""
DeepSpeed QLoRA Inference Script with vLLM

This script demonstrates how to use trained QLoRA models for inference with vLLM acceleration.

python -m workflow.train_text_sql.deepspeed_trainer.scripts.inference
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
from ..utils.other_utils import load_config
import torch
from typing import Optional
try:
    from vllm import LLM, SamplingParams, LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, falling back to transformers")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DeepSpeed QLoRA Inference with vLLM")
    
    # 模型参数
    parser.add_argument(
        "--base_model",
        type=str,
        default="workflow/models/Qwen2.5-7B-Instruct",
        help="基础模型路径或名称"
    )
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="workflow/train_text_sql/deepspeed_trainer/outputs",
        help="LoRA适配器路径"
    )

    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default="workflow/train_text_sql/deepspeed_trainer/configs/train_config.yaml",
        help="推理配置文件路径"
    )
    
    # 推理参数
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大生成长度"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="生成温度"
    )
    
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="是否信任远程代码"
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="基础模型是否支持聊天模式"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="是否批量推理test data并评估准确率"
    )
    
    return parser.parse_args()

def load_model_and_tokenizer(
    base_model: str,
    adapter_path: str,
    trust_remote_code: bool = False
):
    """加载模型"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading base model: {base_model}")
    logger.info(f"Loading adapter: {adapter_path}")
    
    if VLLM_AVAILABLE:
        # 使用vLLM加载基础模型
        model = LLM(
            model=base_model,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            gpu_memory_utilization=0.9
        )
        # vLLM模型也返回tokenizer，但实际不使用
        return model, model.get_tokenizer(), adapter_path
    else:
        # 使用transformers加载模型
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=trust_remote_code,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        
        return model, tokenizer, None

def generate_response(
    model,
    tokenizer=None,
    prompt: str = "",
    max_length: int = 512,
    temperature: float = 0.7,
    adapter_path: Optional[str] = None
) -> str:
    """生成响应"""
    if VLLM_AVAILABLE:
        # 使用vLLM生成
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_length,
            stop=["<|im_end|>", "\n\n"]
        )
        
        # 创建LoRARequest实例
        lora_request = LoRARequest(adapter_path, 1) if adapter_path else None
        
        outputs = model.generate([prompt], sampling_params, lora_request=lora_request)
        generated_text = outputs[0].outputs[0].text.strip()
        return generated_text
    else:
        # 使用transformers生成
        if tokenizer is None:
            raise ValueError("tokenizer is required when using transformers")
            
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()

def instruct_interactive_mode(model, tokenizer, args, adapter_path: Optional[str] = None):
    """交互模式"""
    print("进入交互模式，输入 'quit' 退出")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if not user_input:
                continue
            
            config = load_config(args.config)
            # 格式化输入
            system_prompt = config['data']['format']['system_message']
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            # 生成响应
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                adapter_path=adapter_path
            )
            
            print(f"\n助手: {response}")
        
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            
def chat_interactive_mode(model, tokenizer, args, adapter_path: Optional[str] = None):
    """交互模式"""
    print("进入交互模式，输入 'quit' 退出")
    print("-" * 50)
    history = None
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if not user_input:
                continue
            
            config = load_config(args.config)
            # 格式化输入
            system_prompt = config['data']['format']['system_message']
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            response, history = model.chat(prompt, history)
            
            print(f"\n助手: {response}")
        
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")

def instruct_batch_mode(model, tokenizer, args, adapter_path: Optional[str] = None, test_path="workflow/text2sql_dataset_generator/test_text_sql.json"):
    """批量推理模式：对test data中的每个question生成SQL并评估准确率"""
    import json
    import os
    
    if not os.path.exists(test_path):
        print(f"测试文件不存在: {test_path}")
        return
    
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    print(f"开始批量推理，共 {total_items} 个样本")
    
    for idx, item in enumerate(data, 1):
        conversations = item.get("conversations", [])
        user_msg = next((msg["value"] for msg in conversations if msg["from"] == "user"), None)
        gt_sql = next((msg["value"] for msg in conversations if msg["from"] == "assistant"), None)
        if not user_msg or not gt_sql:
            print(f"[{idx}/{total_items}] 跳过无效样本")
            continue
            
        print(f"\n[{idx}/{total_items}] 处理样本:")
        print(f"Question: {user_msg}")
        print(f"Ground Truth SQL: {gt_sql}")
        
        config = load_config(args.config)
        system_prompt = config['data']['format']['system_message']
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        try:
            pred_sql = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                adapter_path=adapter_path
            )
            print(f"Generated SQL: {pred_sql}")
            
            # 将推理结果写回数据
            conversations.append({
                "from": "predict",
                "value": pred_sql
            })
            
        except Exception as e:
            print(f"推理失败: {e}")
            # 即使失败也添加空的预测结果
            conversations.append({
                "from": "predict",
                "value": ""
            })
    
    # 将更新后的数据写回文件
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n推理完成，结果已写回文件: {test_path}")

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting DeepSpeed QLoRA inference with vLLM")
    
    # 检查GPU
    if torch.cuda.is_available():
        logger.info(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("No GPU found, using CPU")
    
    # 加载模型和分词器
    model, tokenizer, adapter_path = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        trust_remote_code=args.trust_remote_code
    )
    
    print("模型加载完成！")
    if args.batch:
        instruct_batch_mode(model, tokenizer, args, adapter_path)
    elif args.chat:
        # 如果基础模型支持聊天模式
        chat_interactive_mode(model, tokenizer, args, adapter_path)
    else:
        # 否则使用指令模式
        logger.info("基础模型不支持聊天模式，使用指令模式")
        instruct_interactive_mode(model, tokenizer, args, adapter_path)


if __name__ == "__main__":
    main()
