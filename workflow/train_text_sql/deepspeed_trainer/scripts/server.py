#!/usr/bin/env python3
"""
DeepSpeed QLoRA Prediction API with FastAPI

This script provides a FastAPI backend for text-to-SQL prediction using trained QLoRA models.

python -m workflow.train_text_sql.deepspeed_trainer.scripts.server
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
from ..utils.other_utils import load_config
import torch
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    from vllm import LLM, SamplingParams
    import vllm
    LoRARequest = getattr(vllm, 'LoRARequest', None)
    VLLM_AVAILABLE = True
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    LoRARequest = None
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, falling back to transformers")

# 使用transformers框架进行推理
VLLM_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# FastAPI app
app = FastAPI(
    title="Text-to-SQL Prediction API",
    description="API for generating SQL queries from natural language questions",
    version="1.0.0"
)

# 托管静态文件（index.html, main.js, style.css）
static_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
adapter_path = None
config = None

# Pydantic models for API
class QuestionRequest(BaseModel):
    question: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.1

class QuestionResponse(BaseModel):
    question: str
    sql: str
    status: str
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vllm_available: bool

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DeepSpeed QLoRA Prediction API")

    # 基础模型
    parser.add_argument(
        "--base_model",
        type=str,
        default="workflow/models/Qwen2.5-7B-Instruct",
        help="基础模型路径"
    )
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="workflow/train_text_sql/deepspeed_trainer/outputs",
        help="LoRA适配器路径"
    )

    parser.add_argument(
        "--merged_model",
        type=str,
        default=None,
        help="融合后的模型路径（如果提供，将优先使用此路径进行vLLM推理）"
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
    
    # API参数
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API服务器主机地址"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API服务器端口"
    )
    
    return parser.parse_args()

def load_model_and_tokenizer(
    base_model: str,
    adapter_path: str,
    merged_model: Optional[str] = None,
    trust_remote_code: bool = False
):
    """先加载基础模型，再加载适配器权重"""
    logger = logging.getLogger(__name__)
    
    # 如果提供了融合后的模型路径，优先使用
    if merged_model:
        logger.info(f"Loading merged model for vLLM: {merged_model}")
        if VLLM_AVAILABLE:
            # 使用vLLM加载融合后的模型
            model = LLM(
                model=merged_model,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                gpu_memory_utilization=0.9
            )
            return model, model.get_tokenizer()
        else:
            logger.info(f"Loading merged model for transformers: {merged_model}")
            # 使用transformers加载融合后的模型
            tokenizer = AutoTokenizer.from_pretrained(
                merged_model,
                trust_remote_code=trust_remote_code
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                merged_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=trust_remote_code,
                device_map="auto" if torch.cuda.is_available() else None
            )
            model.eval()
            return model, tokenizer
    else:
        if VLLM_AVAILABLE:
            # 使用vLLM加载基础模型
            model = LLM(
                model=base_model,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                gpu_memory_utilization=0.9
            )
            return model, model.get_tokenizer()
        else:
            logger.info(f"Loading base model: {base_model}")
            # 先加载基础模型
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=trust_remote_code,
                device_map="auto" if torch.cuda.is_available() else None
            )
            # 再加载适配器权重
            logger.info(f"Loading adapter: {adapter_path}")
            model = PeftModel.from_pretrained(base_model_obj, adapter_path)
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=trust_remote_code
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            return model, tokenizer

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
        outputs = model.generate([prompt], sampling_params)
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

# API endpoints
@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {
        "message": "Text-to-SQL Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        vllm_available=VLLM_AVAILABLE
    )

@app.post("/predict", response_model=QuestionResponse)
async def predict_sql(request: QuestionRequest):
    """预测SQL"""
    global model, tokenizer, adapter_path, config
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 格式化输入
        if config is None:
            raise HTTPException(status_code=500, detail="Config not loaded")
        system_prompt = config['data']['format']['system_message']
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{request.question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        # 生成响应
        sql = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=request.max_length if request.max_length is not None else 512,
            temperature=request.temperature if request.temperature is not None else 0.1,
            adapter_path=adapter_path
        )
        
        return QuestionResponse(
            question=request.question,
            sql=sql,
            status="success"
        )
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return QuestionResponse(
            question=request.question,
            sql="",
            status="error",
            message=str(e)
        )

@app.post("/chat")
async def chat_endpoint(request: QuestionRequest):
    """聊天接口（与predict相同，但返回格式更友好）"""
    response = await predict_sql(request)
    return {
        "question": response.question,
        "answer": response.sql,
        "status": response.status,
        "message": response.message
    }

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Text-to-SQL Prediction API")
    
    # 检查GPU
    if torch.cuda.is_available():
        logger.info(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("No GPU found, using CPU")
    
    # 加载配置
    global config
    config = load_config(args.config)
    
    # 加载模型和分词器
    global model, tokenizer
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        merged_model=args.merged_model,
        trust_remote_code=args.trust_remote_code
    )
    
    logger.info("Model loaded successfully!")
    logger.info(f"API server starting on {args.host}:{args.port}")
    
    # 启动FastAPI服务器
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    main() 