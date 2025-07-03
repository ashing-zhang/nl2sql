"""
Custom model implementations for DeepSpeed QLoRA training.
"""

import logging
from typing import Dict, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


def create_custom_model(
    model_name: str,
    model_type: str,
    lora_config: Dict,
    quantization_config: Optional[Dict] = None,
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, Dict]] = None
):
    """创建自定义模型"""
    logger.info(f"Creating custom model: {model_name} ({model_type})")
    
    if model_type == "llama":
        return create_llama_model(model_name, lora_config, quantization_config, trust_remote_code, torch_dtype, device_map)
    elif model_type == "mistral":
        return create_mistral_model(model_name, lora_config, quantization_config, trust_remote_code, torch_dtype, device_map)
    elif model_type == "qwen":
        return create_qwen_model(model_name, lora_config, quantization_config, trust_remote_code, torch_dtype, device_map)
    elif model_type == "chatglm":
        return create_chatglm_model(model_name, lora_config, quantization_config, trust_remote_code, torch_dtype, device_map)
    else:
        logger.warning(f"Unknown model type: {model_type}, using base model creation")
        from .base_model import create_base_model
        return create_base_model(model_name, lora_config, quantization_config, trust_remote_code, torch_dtype, device_map)


def create_llama_model(
    model_name: str,
    lora_config: Dict,
    quantization_config: Optional[Dict] = None,
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, Dict]] = None
):
    """创建LLaMA模型"""
    logger.info(f"Creating LLaMA model: {model_name}")
    
    lora_config.setdefault("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    return create_base_model_with_config(
        model_name, lora_config, quantization_config, trust_remote_code, torch_dtype, device_map
    )


def create_mistral_model(
    model_name: str,
    lora_config: Dict,
    quantization_config: Optional[Dict] = None,
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, Dict]] = None
):
    """创建Mistral模型"""
    logger.info(f"Creating Mistral model: {model_name}")
    
    lora_config.setdefault("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    return create_base_model_with_config(
        model_name, lora_config, quantization_config, trust_remote_code, torch_dtype, device_map
    )


def create_qwen_model(
    model_name: str,
    lora_config: Dict,
    quantization_config: Optional[Dict] = None,
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, Dict]] = None
):
    """创建Qwen模型"""
    logger.info(f"Creating Qwen model: {model_name}")
    
    lora_config.setdefault("target_modules", [
        "c_attn", "c_proj", "w1", "w2"
    ])
    
    return create_base_model_with_config(
        model_name, lora_config, quantization_config, trust_remote_code, torch_dtype, device_map
    )


def create_chatglm_model(
    model_name: str,
    lora_config: Dict,
    quantization_config: Optional[Dict] = None,
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, Dict]] = None
):
    """创建ChatGLM模型"""
    logger.info(f"Creating ChatGLM model: {model_name}")
    
    lora_config.setdefault("target_modules", [
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"
    ])
    
    return create_base_model_with_config(
        model_name, lora_config, quantization_config, trust_remote_code, torch_dtype, device_map
    )


def create_base_model_with_config(
    model_name: str,
    lora_config: Dict,
    quantization_config: Optional[Dict] = None,
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, Dict]] = None
):
    """使用配置创建基础模型"""
    from .base_model import create_base_model
    return create_base_model(
        model_name, lora_config, quantization_config, trust_remote_code, torch_dtype, device_map
    )


def get_model_specific_config(model_type: str) -> Dict:
    """获取模型特定的配置"""
    configs = {
        "llama": {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none"
        },
        "mistral": {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none"
        },
        "qwen": {
            "target_modules": ["c_attn", "c_proj", "w1", "w2"],
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none"
        },
        "chatglm": {
            "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none"
        }
    }
    
    return configs.get(model_type, {}) 