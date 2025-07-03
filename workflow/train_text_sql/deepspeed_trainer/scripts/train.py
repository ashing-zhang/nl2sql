#!/usr/bin/env python3
"""
DeepSpeed QLoRA Training Script

This script demonstrates how to use DeepSpeed for distributed QLoRA training.

如果训练数据不是位于deepspeed_trainer/data文件夹中（比如位于text2sql_dataset_generator文件夹中），则可从根目录finance_QA启动训练，
启动命令为：
deepspeed --module workflow.train_text_sql.deepspeed_trainer.scripts.train

"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from ..utils.other_utils import load_config
import torch
from transformers import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from ..utils.data_utils import TextSqlDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_utils import load_dataset, preprocess_data, create_data_collator, process_conversations_data
from utils.model_utils import setup_tokenizer, create_model, setup_training_args, setup_logging
from build_models.base_model import create_base_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DeepSpeed QLoRA Training")
    
    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default="workflow/train_text_sql/deepspeed_trainer/configs/train_config.yaml",
        help="训练配置文件路径"
    )
    
    # DeepSpeed配置
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="workflow/train_text_sql/deepspeed_trainer/configs/deepspeed_config.json",
        help="DeepSpeed配置文件路径"
    )
    
    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        help="模型名称（覆盖配置文件）"
    )
    
    # 数据参数
    parser.add_argument(
        "--train_file",
        type=str,
        help="训练数据文件路径（覆盖配置文件）"
    )
    
    # 训练参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="workflow/train_text_sql/deepspeed_trainer/outputs",
        help="输出目录"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="训练轮数（覆盖配置文件）"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help="批次大小（覆盖配置文件）"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="学习率（覆盖配置文件）"
    )
    
    # 其他参数
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="本地GPU排名"
    )
    
    return parser.parse_args()

def update_config_with_args(config: dict, args) -> dict:
    """使用命令行参数更新配置"""
    if args.model_name:
        config['model']['base_model'] = args.model_name
    
    if args.train_file:
        config['data']['train_file'] = args.train_file
    
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(
        project_name="deepspeed-qlora",
        log_to_wandb=False,
        log_to_tensorboard=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting DeepSpeed QLoRA training")
    
    # 加载配置
    config = load_config(args.config)
    # config = update_config_with_args(config, args)
    
    # 设置随机种子
    import random
    import numpy as np
    seed = config.get('advanced', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 检查GPU
    if torch.cuda.is_available():
        logger.info(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("No GPU found, using CPU")

    # 显示是否使用所有GPU
    num_gpus = torch.cuda.device_count()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if num_gpus > 1:
        logger.info(f"多卡训练: 检测到 {num_gpus} 张GPU。")
        if cuda_visible_devices:
            logger.info(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}，实际可用GPU数量: {num_gpus}")
        else:
            logger.info("未设置CUDA_VISIBLE_DEVICES，默认使用所有可用GPU。")
    elif num_gpus == 1:
        logger.info("单卡训练: 仅检测到1张GPU。")
    else:
        logger.warning("未检测到可用GPU，将使用CPU进行训练。")
    
    # 设置分词器
    tokenizer = setup_tokenizer(
        model_name=config['model']['base_model'],
        trust_remote_code=config['model'].get('trust_remote_code', False)
    )
    
    train_dataset = TextSqlDataset(config,'train',tokenizer)
    val_dataset = TextSqlDataset(config,'val',tokenizer)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # 提升GPU计算效率
    )
    # 创建模型
    model = create_base_model(
        model_name=config['model']['base_model'],
        lora_config=config['model']['lora'],
        quantization_config=config['model']['quantization'],
        trust_remote_code=config['model'].get('trust_remote_code', False)
    )
    
    # 设置训练参数
    training_args = setup_training_args(config, args.output_dir)
    
    # 添加DeepSpeed配置
    if hasattr(args, "deepspeed") and args.deepspeed is not None:
        training_args.deepspeed = args.deepspeed
    # logger.info(f"training_args:{training_args}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator
    )
    
    # 开始训练
    logger.info("Starting training...")
    trainer.train()
    
    # 保存最终模型
    logger.info("Saving final model...")
    # 保存LoRA适配器权重
    trainer.save_model()
    
    # 可选：保存完整模型（包含基础模型权重）
    # trainer.save_model(safe_serialization=True, max_shard_size="2GB")
    
    # 保存分词器
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 