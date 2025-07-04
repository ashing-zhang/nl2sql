"""
Base model utilities for DeepSpeed QLoRA training.
"""

import logging
from typing import Dict, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


def create_base_model(
    model_name: str,
    lora_config: Dict,
    quantization_config: Optional[Dict] = None,
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, Dict]] = None
):
    """
    创建基础模型。
    
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
    logger.info(f"Creating base model: {model_name}")
    
    # 加载模型配置
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # 设置量化配置
    if quantization_config:
        from transformers.utils.quantization_config import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_config.get("load_in_4bit", False),
            load_in_8bit=quantization_config.get("load_in_8bit", False),
            bnb_4bit_compute_dtype=quantization_config.get("bnb_4bit_compute_dtype", torch.float16),
            bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", False),
            bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4")
        )
    else:
        bnb_config = None
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    # 打印模型参数名
    # logger.info(f"Model parameters: {[name for name, _ in model.named_parameters()]}")
    
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
    
    logger.info("Base model creation completed")
    return model


def get_model_config(model_name: str, trust_remote_code: bool = False) -> Dict:
    """
    获取模型配置信息。
    
    Args:
        model_name: 模型名称
        trust_remote_code: 是否信任远程代码
    
    Returns:
        Dict: 模型配置信息
    """
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    return {
        "model_type": config.model_type,
        "vocab_size": config.vocab_size,
        "hidden_size": getattr(config, 'hidden_size', None),
        "num_attention_heads": getattr(config, 'num_attention_heads', None),
        "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
        "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
        "architectures": getattr(config, 'architectures', None)
    } 