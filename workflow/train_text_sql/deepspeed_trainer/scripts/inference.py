#!/usr/bin/env python3
"""
DeepSpeed QLoRA Inference Script

This script demonstrates how to use trained QLoRA models for inference.

python -m workflow.train_text_sql.deepspeed_trainer.scripts.inference
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
from ..utils.other_utils import load_config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DeepSpeed QLoRA Inference")
    
    # 模型参数
    parser.add_argument(
        "--base_model",
        type=str,
        default="workflow/models/Tongyi-Finance-14B-Chat",
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
    
    return parser.parse_args()


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: str,
    trust_remote_code: bool = False
):
    """加载模型和分词器"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading base model: {base_model}")
    logger.info(f"Loading adapter: {adapter_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7
) -> str:
    """生成响应"""
    # 编码输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # 移动到设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
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
    
    # 解码输出
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text.strip()

def instruct_interactive_mode(model, tokenizer, args):
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
                temperature=args.temperature
            )
            
            print(f"\n助手: {response}")
        
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            
def chat_interactive_mode(model, tokenizer, args):
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
            
            response, history = model.chat(tokenizer, prompt, history)
            
            print(f"\n助手: {response}")
        
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting DeepSpeed QLoRA inference")
    
    # 检查GPU
    if torch.cuda.is_available():
        logger.info(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("No GPU found, using CPU")
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        trust_remote_code=args.trust_remote_code
    )
    
    print("模型加载完成！")
    if args.chat:
        # 如果基础模型支持聊天模式
        chat_interactive_mode(model, tokenizer, args)
    else:
        # 否则使用指令模式
        logger.info("基础模型不支持聊天模式，使用指令模式")
        instruct_interactive_mode(model, tokenizer, args)


if __name__ == "__main__":
    main()
