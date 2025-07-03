#!/usr/bin/env python3
"""
Quick Start Script for DeepSpeed QLoRA Training

This script demonstrates the basic usage of the project.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.data_utils import load_dataset, get_dataset_info
from utils.model_utils import setup_tokenizer, get_compute_dtype
from models.base_model import get_model_config


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("DeepSpeed QLoRA Training - Quick Start")
    
    # 1. 加载数据集
    logger.info("1. Loading dataset...")
    try:
        dataset = load_dataset(
            data_file="data/sample_data.json",
            validation_split=0.2,
            format_type="instruction"
        )
        
        # 显示数据集信息
        train_info = get_dataset_info(dataset['train'])
        val_info = get_dataset_info(dataset['validation'])
        
        logger.info(f"Training set: {train_info['num_examples']} examples")
        logger.info(f"Validation set: {val_info['num_examples']} examples")
        
        if 'text_length_stats' in train_info:
            stats = train_info['text_length_stats']
            logger.info(f"Text length - Mean: {stats['mean']:.1f}, Max: {stats['max']}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # 2. 设置分词器
    logger.info("2. Setting up tokenizer...")
    try:
        model_name = "microsoft/DialoGPT-medium"
        tokenizer = setup_tokenizer(model_name)
        logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        
    except Exception as e:
        logger.error(f"Failed to setup tokenizer: {e}")
        return
    
    # 3. 获取模型配置
    logger.info("3. Getting model configuration...")
    try:
        config = get_model_config(model_name)
        logger.info(f"Model type: {config['model_type']}")
        logger.info(f"Vocab size: {config['vocab_size']}")
        logger.info(f"Hidden size: {config['hidden_size']}")
        
    except Exception as e:
        logger.error(f"Failed to get model config: {e}")
        return
    
    # 4. 检查计算环境
    logger.info("4. Checking computation environment...")
    import torch
    
    if torch.cuda.is_available():
        logger.info(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("No GPU found, using CPU")
    
    # 5. 显示训练命令
    logger.info("5. Training commands:")
    print("\n" + "="*60)
    print("TRAINING COMMANDS")
    print("="*60)
    print("# Single GPU training:")
    print("python scripts/train.py")
    print()
    print("# Multi-GPU training:")
    print("deepspeed --num_gpus=2 scripts/train.py --deepspeed configs/deepspeed_config.json")
    print()
    print("# Custom model and data:")
    print("python scripts/train.py --model_name microsoft/DialoGPT-medium --train_file data/sample_data.json")
    print()
    print("# Inference:")
    print("python scripts/inference.py --base_model microsoft/DialoGPT-medium --adapter_path outputs/checkpoint-xxx")
    print("="*60)
    
    logger.info("Quick start completed successfully!")
    logger.info("Check the README.md for detailed usage instructions.")


if __name__ == "__main__":
    main() 