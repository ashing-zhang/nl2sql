"""
Model utilities for DeepSpeed QLoRA training.
"""

import logging
import os
from typing import Dict, Optional, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


def setup_tokenizer(
    model_name: str,
    trust_remote_code: bool = False,
    padding_side: str = "right",
    use_fast: bool = True
) -> AutoTokenizer:
    """
    设置分词器。
    
    Args:
        model_name: 模型名称
        trust_remote_code: 是否信任远程代码
        padding_side: 填充方向
        use_fast: 是否使用快速分词器
    
    Returns:
        AutoTokenizer: 配置好的分词器
    """
    logger.info(f"Setting up tokenizer for {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast
    )
    
    # 设置填充token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置填充方向
    tokenizer.padding_side = padding_side
    
    logger.info(f"Tokenizer setup completed. Vocab size: {tokenizer.vocab_size}")
    return tokenizer


def get_compute_dtype(compute_dtype: str = "float16") -> torch.dtype:
    """
    获取计算数据类型。
    
    Args:
        compute_dtype: 计算数据类型字符串
    
    Returns:
        torch.dtype: PyTorch数据类型
    """
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    
    if compute_dtype not in dtype_map:
        raise ValueError(f"Unsupported compute dtype: {compute_dtype}")
    
    return dtype_map[compute_dtype]


def create_model(
    model_name: str,
    lora_config: Dict,
    quantization_config: Optional[Dict] = None,
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, Dict]] = None
):
    """
    创建模型。
    
    Args:
        model_name: 模型名称
        lora_config: LoRA配置
        quantization_config: 量化配置
        trust_remote_code: 是否信任远程代码
        torch_dtype: PyTorch数据类型
        device_map: 设备映射
    
    Returns:
        配置好的模型
    """
    logger.info(f"Creating model: {model_name}")
    
    # 设置量化配置
    bnb_config = None
    if quantization_config:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_config.get("load_in_4bit", False),
            load_in_8bit=quantization_config.get("load_in_8bit", False),
            bnb_4bit_compute_dtype=get_compute_dtype(
                quantization_config.get("bnb_4bit_compute_dtype", "float16")
            ),
            bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", False),
            bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4")
        )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    
    # 为量化训练准备模型
    if quantization_config and (quantization_config.get("load_in_4bit") or quantization_config.get("load_in_8bit")):
        model = prepare_model_for_kbit_training(model)
    
    # 设置LoRA配置
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_config.get("dropout", 0.1),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM")
    )
    
    # 应用LoRA
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    logger.info("Model creation completed")
    return model


def setup_training_args(
    config: Dict,
    output_dir: str = "outputs"
) -> TrainingArguments:
    """
    设置训练参数。
    
    Args:
        config: 配置字典
        output_dir: 输出目录
    
    Returns:
        TrainingArguments: 训练参数
    """
    training_config = config.get("training", {})
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("batch_size", 4),
        per_device_eval_batch_size=training_config.get("batch_size", 4),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps", 100),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        eval_steps=training_config.get("eval_steps", 500),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=training_config.get("fp16", True),
        bf16=training_config.get("bf16", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        dataloader_num_workers=config.get("hardware", {}).get("dataloader_num_workers", 4),
        dataloader_pin_memory=config.get("hardware", {}).get("dataloader_pin_memory", True),
        remove_unused_columns=config.get("advanced", {}).get("remove_unused_columns", True),
        group_by_length=config.get("advanced", {}).get("group_by_length", False),
        report_to=["tensorboard"] if config.get("logging", {}).get("log_to_tensorboard", True) else [],
        run_name=config.get("logging", {}).get("run_name"),
        seed=config.get("advanced", {}).get("seed", 42),
        deepspeed=config.get("deepspeed", {}).get("config_file", {})
    )
    
    return training_args


def setup_logging(
    project_name: str = "deepspeed-qlora",
    log_to_wandb: bool = False,
    log_to_tensorboard: bool = True
):
    """
    设置日志记录。
    
    Args:
        project_name: 项目名称
        log_to_wandb: 是否使用WandB
        log_to_tensorboard: 是否使用TensorBoard
    """
    import logging
    
    # 设置基本日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    # 设置WandB
    if log_to_wandb:
        try:
            import wandb
            wandb.init(project=project_name)
            logger.info("WandB logging enabled")
        except ImportError:
            logger.warning("WandB not installed, skipping WandB logging")
    
    # 设置TensorBoard
    if log_to_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir='runs')
            logger.info("TensorBoard logging enabled")
        except ImportError:
            logger.warning("TensorBoard not installed, skipping TensorBoard logging")


def get_model_size_info(model) -> Dict:
    """
    获取模型大小信息。
    
    Args:
        model: 模型
    
    Returns:
        Dict: 模型大小信息
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "param_size_mb": param_size / 1024**2,
        "buffer_size_mb": buffer_size / 1024**2,
        "total_size_mb": size_all_mb
    }


def save_model_info(model, output_dir: str):
    """
    保存模型信息。
    
    Args:
        model: 模型
        output_dir: 输出目录
    """
    import json
    
    info = {
        "model_size": get_model_size_info(model),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters())
    }
    
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Model info saved to {output_dir}/model_info.json") 