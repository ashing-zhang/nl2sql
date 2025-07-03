'''
FastAPI wrapper for SQL generation using Tongyi-Finance-14B model
'''

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import sqlite3
import argparse
import torch
from transformers.trainer_utils import set_seed
from vllm import SamplingParams, AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
import uvicorn
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize on startup
    if not initialize_globals():
        raise RuntimeError("Failed to initialize application globals")
    yield
    # Cleanup on shutdown
    if conn:
        conn.close()

app = FastAPI(
    title="SQL Generation API",
    description="API for generating SQL queries using Tongyi-Finance-14B model",
    lifespan=lifespan
)

class SQLRequest(BaseModel):
    prompt: str
    execute: Optional[bool] = False

class SQLResponse(BaseModel):
    sql_query: str
    execution_result: Optional[list] = None
    error: Optional[str] = None

engine = None
lora_request = None
conn = None
system_message = "You are a helpful assistant that translates natural language to SQL queries."
SYSTEM_MESSAGE_REQUEST_ID = "system_sql"

def initialize_engine(args):
    engine_args = AsyncEngineArgs(
        model=args.model_name_path,
        enable_lora=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        quantization="awq",
        max_num_seqs=256
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    lora_request = LoRARequest(
        "sql_lora",
        1,
        args.lora_path
    )
    
    return engine, lora_request

async def generate_sql(prompt: str) -> str:
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    # Generate full context including system message
    input_text = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    sampling_params = SamplingParams(
        temperature=0.1,
        top_k=50,
        top_p=0.95,
        max_tokens=512
    )
    
    try:
        # Handle AsyncGenerator output
        outputs = []
        # Use simple request_id since we're not doing special caching
        request_id = f"sql_{id(outputs)}"
        async for output in engine.generate(
            prompt=input_text,
            sampling_params=sampling_params,
            lora_request=lora_request,
            request_id=request_id
        ):
            outputs.append(output)
        
        if not outputs:
            raise HTTPException(status_code=500, detail="No output generated")
            
        return outputs[-1].outputs[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

async def execute_sql(sql_query: str) -> tuple:
    if conn is None:
        raise HTTPException(status_code=500, detail="Database not connected")
        
    try:
        cur = conn.cursor()
        cur.execute(sql_query)
        return cur.fetchall(), None
    except Exception as e:
        if '交易日期' in sql_query:
            alt_query = sql_query.replace('交易日期', '交易日')
        elif '交易日' in sql_query:
            alt_query = sql_query.replace('交易日', '交易日期')
        else:
            return None, str(e)
        
        try:
            cur.execute(alt_query)
            return cur.fetchall(), None
        except Exception as e2:
            return None, str(e2)

@app.post("/generate-sql", response_model=SQLResponse)
async def generate_sql_endpoint(request: SQLRequest):
    try:
        sql_query = await generate_sql(request.prompt)
        sql_query = sql_query.replace("”", '').replace("“", '')
        
        execution_result = None
        error = None
        if request.execute:
            execution_result, error = await execute_sql(sql_query)
        
        return SQLResponse(
            sql_query=sql_query,
            execution_result=execution_result,
            error=error
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_path", type=str, required=True,
                      help="Base model path")
    parser.add_argument("--lora_path", type=str, required=True,
                      help="LoRA adapter path") 
    parser.add_argument("--db_path", type=str, required=True,
                      help="Database path")
    parser.add_argument("--port", type=int, default=8000,
                      help="API server port")
    return parser.parse_args()

def initialize_globals():
    global engine, lora_request, conn
    try:
        args = parse_args()
        set_seed(42)
        engine, lora_request = initialize_engine(args)
        conn = sqlite3.connect(args.db_path)
        print("Initialization completed successfully")
        return True
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        return False


if __name__ == '__main__':
    if initialize_globals():
        args = parse_args()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        print("Cannot start server due to initialization errors")
