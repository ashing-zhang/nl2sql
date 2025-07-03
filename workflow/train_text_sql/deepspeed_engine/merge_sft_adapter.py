import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge(BASE_MODEL_PATH,SFT_ADAPTER_PATH):
    # --- 1. 加载基础模型和 SFT 阶段训练好的 LoRA 适配器 ---
    print(f"\n--- 阶段 1: 加载基础模型和 SFT 适配器 ---")

    # 1.1 首先，以全精度或半精度加载原始基础模型
    # 注意：这里不使用 BitsAndBytesConfig 进行 4-bit 量化，因为我们要合并到这个模型上
    # 融合后的模型精度将取决于这里的 torch_dtype
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16, # 假设您的原始模型是 bfloat16 或你想融合为 bfloat16
        device_map="auto", # 自动分配到GPU
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    print(f"原始基础模型 '{BASE_MODEL_PATH}' 已加载为 {base_model.dtype} 精度。")

    # 1.2 加载 SFT 阶段训练好的 LoRA 适配器
    # 注意：这里使用 PeftModel.from_pretrained，它会将适配器附加到已加载的 base_model 上
    model_with_sft_adapter = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    print(f"SFT 适配器已从 '{SFT_ADAPTER_PATH}' 加载并附加到模型。")

    # --- 2. 将 LoRA 适配器权重融合到基础模型中 ---

    print(f"\n--- 阶段 2: 融合 LoRA 适配器 ---")

    # 2.1 使用 merge_and_unload() 方法融合适配器
    # 这个方法会返回一个新的 PreTrainedModel 实例，其权重已包含适配器的修改
    # 并且会卸载 PEFT 适配器，使模型变回标准的 transformers 模型
    merged_model = model_with_sft_adapter.merge_and_unload()
    print("LoRA 适配器已成功融合到基础模型中。")
    # 融合后模型的精度: torch.bfloat16
    print(f"融合后模型的精度: {merged_model.dtype}")

    # 2.2 保存融合后的模型
    # 这一步很重要，因为您需要保存这个融合后的模型，以便后续加载和再次量化
    merged_model.save_pretrained(MERGED_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH) # 也要保存tokenizer
    print(f"融合后的模型已保存到: {MERGED_MODEL_SAVE_PATH}")

    # 清理内存
    '''
        torch.cuda.empty_cache() 的作用是强制 PyTorch 
        释放当前未被任何张量引用的、但仍在缓存中的 GPU 内存。
        它不会释放正在被使用的内存。
    '''
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 定义基础模型和 SFT 适配器的路径
    BASE_MODEL_PATH = "../models/Qwen-7B-Chat"  
    SFT_ADAPTER_PATH = "./model_save/sql_lora"  
    MERGED_MODEL_SAVE_PATH = "./model_save/merged_sft_sql_model"

    merge(BASE_MODEL_PATH, SFT_ADAPTER_PATH)