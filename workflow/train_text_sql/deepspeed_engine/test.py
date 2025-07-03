'''
    1.RTX 3090 * 1卡可以顺利对BitsAndBytesConfig 4比特量化的14B模型进行推理
    2.但BitsAndBytesConfig 4比特量化的14B模型甚至都不能生成sql语句
    3.如果使用RTX 3090 * 2卡对模型进行非量化推理，速度极慢，且生成的sql体现出lora权重完全没有起作用
    python -m workflow.train_text_sql.deepspeed_engine.test
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import get_peft_model, PeftModel
import sqlite3
import os
import argparse
import torch
import gc
import torch.distributed as dist
from transformers import BitsAndBytesConfig, set_seed
from accelerate import Accelerator
import torch
from safetensors.torch import load_file as safe_load_file, save_file

# 4比特量化配置
def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,  # 启用4比特量化
        bnb_4bit_use_double_quant=True,  # 嵌套量化，额外节省0.4位
        bnb_4bit_quant_type="nf4",  # 使用NF4量化类型
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用bfloat16
    )

def merge_model(args):
    # 初始化加速器
    accelerator = Accelerator()
    
    # 量化配置
    quantization_config = get_quantization_config()
    
    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_path,
        trust_remote_code=True,
        quantization_config=quantization_config,  # 应用4比特量化
        device_map="auto"  # 自动分配多GPU
    )

    lora_path = os.path.join(args.lora_dir, args.adapter_name)
    
    model = PeftModel.from_pretrained(model, lora_path, adapter_name=args.adapter_name,)
    model = model.merge_and_unload()

    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_path,
        trust_remote_code=True
    )
    
    # 准备模型和tokenizer
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # 设置为评估模式
    model.eval()
    
    print("量化模型加载完成，内存占用:", 
          torch.cuda.memory_allocated()/1024**2, "MB")
    
    return model, tokenizer

def generate_sql(model, tokenizer, prompt, system_message):
    # 生成SQL
    response, history = model.chat(tokenizer, prompt, history=None,system=system_message)
    return response

def test_sql_lora(args):
    model, tokenizer = merge_model(args)
    model.eval()  # 设置为评估模式
    
    conn = sqlite3.connect(args.db_path)
    system_message = "You are a helpful assistant that translates natural language to SQL queries."
    
    while True:
        print("\n用户输入SQL生成诉求：", end="")
        prompt = input().strip()
        if not prompt:
            continue
            
        with torch.no_grad():
            # 生成SQL查询
            response = generate_sql(model, tokenizer, prompt, system_message)
        
        print("当前LORA模型生成SQL语句为：", response)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_path", type=str, default='workflow/models/Qwen2.5-7B-Instruct', 
                        help="模型名称或路径")
    parser.add_argument("--lora_dir", type=str, default='workflow/train_text_sql/deepspeed_engine/model_save', 
                        help="LoRA模型路径")
    parser.add_argument("--adapter_name", type=str, default='sql_lora', 
                        help="LoRA适配器名称")
    parser.add_argument("--db_path", type=str, default='data/dataset/fund_data.db', 
                        help="数据库路径")
    parser.add_argument("--gen_len", type=int, default=128,  
                        help="生成的最大token数")
    parser.add_argument("--seed", type=int, default=42, 
                        help="随机种子")
    args = parser.parse_args()
    
    set_seed(args.seed)  # 设置随机种子
    test_sql_lora(args)
