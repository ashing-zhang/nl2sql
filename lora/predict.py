'''
    使用经过nl2sql数据微调优化过的模型对question.json对../data/data_query.txt中的
    nl查询进行推理，生成对应的sql语句保存在predict_sql.json文件中：
'''
import json
import torch
from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import AutoModelForCausalLM, AutoTokenizer
import tqdm
from config import TrainingConfig

def load_model(base_model_path, lora_weights_path):
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        torch_dtype=torch.float16
    )
    return model

def generate_sql(model, tokenizer, questions, output_path, table_schema):
    results = []
    for item in tqdm.tqdm(questions, desc="Generating SQL"):
        # 构建Alpaca格式输入
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
            Generate only the SQL query in ONE LINE for: {item['question']}
            ### Database Schema:
            {table_schema}
            ### Important Notes:
            1. SQL must be in single line without line breaks
            2. Use space instead of newline between clauses
            3. Do not output the reasoning or thought process, just provide the executable SQL query
            ### Response:
            """
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成SQL
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=250,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 解析输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql = response.split("### Response:")[1].strip()
        
        results.append({
            "question": item["question"],
            "sql": sql
        })

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    config = TrainingConfig()
    # 配置路径
    base_model = config.model_name  # 原始基座模型路径
    lora_weights = "./checkpoints/best_model_sft"  # LoRA权重路径
    input_file = "./data/data_query.txt"   
    output_file = "./data/predict_sql.json"   
    db_schema_path = "../data/dataset/db_schema.json"   # lora dir的上一级dir中的data dir

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = load_model(base_model, lora_weights)
    
    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = [{"question": line.strip()} for line in f]
    # 数据库模式
    table_schema = json.load(open(db_schema_path, "r", encoding="utf-8"))
    # 生成SQL
    generate_sql(model, tokenizer, questions, output_file, table_schema)