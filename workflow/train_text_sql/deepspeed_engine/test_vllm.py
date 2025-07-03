'''
    vLLM version of SQL generation using Tongyi-Finance-14B model
    python -m workflow.train_text_sql.deepspeed_engine.test_vllm
'''

import uuid
from transformers import AutoTokenizer
import sqlite3
import os
import argparse
import torch
import gc
import torch.distributed as dist
from transformers.trainer_utils import set_seed
from vllm import LLM, SamplingParams, AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
from transformers import BitsAndBytesConfig

def initialize_engine(args):
    # 1. Initialize async engine with proper configuration
    engine_args = AsyncEngineArgs(
        model=args.model_name_path,
        enable_lora=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        quantization="bitsandbytes",
        max_num_seqs=256,
        max_model_len=500
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Load LoRA adapter
    lora_request = LoRARequest(
        "sql_lora",  # LoRA name
        1,           # Version
        args.lora_path  # Path to LoRA adapter
    )
    
    return engine, lora_request

async def generate_sql(engine, prompt, system_message, lora_request):
    # Build input with system message and user prompt
    input_text = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0.1,
        top_k=50,
        top_p=0.95,
        max_tokens=512
    )

    # Generate output with LoRA request
    output_generator = engine.generate(
        input_text,
        sampling_params,
        lora_request=lora_request,
        request_id=str(uuid.uuid4())
    )
    # Get the first output from the generator
    try:
        async for output in output_generator:
            if output.outputs and len(output.outputs) > 0:
                return output.outputs[0].text
        return ""  # Return empty string if no output
    except Exception as e:
        print(f"Error during generation: {e}")
        return ""

async def test_sql_lora(args):
    # Initialize async engine and get LoRA request
    engine, lora_request = initialize_engine(args)
    
    conn = sqlite3.connect(args.db_path)
    system_message = "You are a helpful assistant that translates natural language to SQL queries."
    
    while True:
        print("\n用户输入SQL生成诉求：", end="")
        prompt = input().strip()
        if not prompt:
            continue
            
        # Generate SQL using async engine with LoRA
        response = await generate_sql(engine, prompt, system_message, lora_request)
        
        print("当前vLLM模型生成SQL语句为：", response)
        response = response.replace("”", '').replace("“", '')
        
        # 执行SQL查询
        cur = conn.cursor()
        print('执行SQL:', response)
        try:
            cur.execute(response)
            sql_answer = cur.fetchall()
        except Exception as e:
            # 检查并替换 '交易日' 和 '交易日期'
            if '交易日期' in response:
                alt_response = response.replace('交易日期', '交易日')
            elif '交易日' in response:
                alt_response = response.replace('交易日', '交易日期')
            else:
                print(f"SQL执行失败: {e}")
                continue
            print('尝试替换后的sql: ' + alt_response)
            try:
                cur.execute(alt_response)
                sql_answer = cur.fetchall()
                response = alt_response  # 更新为成功的SQL
            except Exception as e2:
                print(f"SQL执行仍然失败: {e2}")
                continue
        print('当前SQL语句查询结果：', sql_answer)
             

if __name__ == '__main__':
    import asyncio
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_path", type=str, default='workflow/models/Tongyi-Finance-14B-Chat', 
                        help="模型名称或路径")
    parser.add_argument("--lora_path", type=str, default='workflow/train_text_sql/deepspeed_engine/model_save/sql_lora', 
                        help="LoRA模型路径")
    parser.add_argument("--db_path", type=str, default='data/dataset/fund_data.db', 
                        help="数据库路径")
    parser.add_argument("--gen_len", type=int, default=128,  
                        help="生成的最大token数")
    parser.add_argument("--seed", type=int, default=42, 
                        help="随机种子")
    args = parser.parse_args()
    
    set_seed(args.seed)  # 设置随机种子
    asyncio.run(test_sql_lora(args))
