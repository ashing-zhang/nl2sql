from peft import LoraConfig
import torch
from transformers import AutoTokenizer

# 配置参数
class Config:
    seed = 42
    model_name = "workflow/models/Qwen2.5-7B-Instruct"    # lora微调该模型
    train_data_dir = "workflow/text2sql_dataset_generator"
    train_json_path = "train_text_sql.json"
    # train_json_path = "train_text_sql_add.json"
    val_data_dir = "workflow/text2sql_dataset_generator"
    val_json_path = "val_text_sql.json"
    # val_json_path = "val_text_sql_add.json"
    test_data_dir = "workflow/text2sql_dataset_generator"
    test_json_path = "test_text_sql.json"
    sql_model_save_path = "workflow/train_text_sql/deepspeed_engine/model_save"
    train_loss_plot_path = "workflow/train_text_sql/deepspeed_engine/train_loss_plot.png"
    val_loss_plot_path = "workflow/train_text_sql/deepspeed_engine/val_loss_plot.png"
    epochs = 5  # 最大epoch数
    early_stop_patience = 25  # 早停耐心值(不能设置太小，不然会导致模型训练不充分)
    improvement_ratio = 0.8  # 早停改善比例(不能设置太小，不然会导致模型训练不充分)
    lora_config = LoraConfig(
        r=32,  # 低秩矩阵的秩
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none", 
        target_modules = [
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ]
    )
    adapter_name = "sql_lora"  # LoRA适配器名称
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    max_len = 512
    system_message = "You are a helpful assistant that translates natural language to SQL queries. " 
    

