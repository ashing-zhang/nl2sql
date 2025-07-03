import json
import torch
from peft import PeftModel
from modelscope import AutoModelForCausalLM, AutoTokenizer
import tqdm
from config import TrainingConfig

def load_model(base_model_path, lora_weights_path):
    # 加载基础模型（未修改）
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载CoT微调的LoRA适配器 # Modified
    model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        torch_dtype=torch.float16
    )
    return model

def generate_sql(model, tokenizer, questions, output_path, table_schema):
    results = []
    for item in tqdm.tqdm(questions, desc="Generating SQL"):
        # 构建支持思维链的提示模板 # Modified
        prompt = f"""### Instruction:
                请分析问题并生成SQL。数据库结构：{table_schema}
                问题：{item['question']}

                ### CoT:
                （模型应在此处生成推理步骤）

                ### Response:
                SELECT..."""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 扩展生成长度以容纳推理过程 # Modified
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=256,  # Modified
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 提取最终答案部分 # Modified
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("response:", response)
        if "### Response:" in response:
            sql_part = response.split("### Response:")[1]
            sql = sql_part.split(";")[0].split("\n")[0].strip()
        else:
            sql = "ERROR: 未检测到有效SQL"
        
        results.append({
            "question": item["question"],
            "sql": sql
        })

    # 保存结果（未修改）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    config = TrainingConfig()
    # 配置路径修改 # Modified
    base_model = config.model_name  
    lora_weights = "./checkpoints/best_model_cot"  # Modified
    input_file = "./data/data_query.txt"   
    output_file = "./data/predict_sql.json"   
    db_schema_path = "../data/dataset/db_schema.json"

    # 模型加载（未修改）
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = load_model(base_model, lora_weights)
    
    # 数据加载（未修改）
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = [{"question": line.strip()} for line in f]
    table_schema = json.load(open(db_schema_path, "r", encoding="utf-8"))
    
    generate_sql(model, tokenizer, questions, output_file, table_schema)