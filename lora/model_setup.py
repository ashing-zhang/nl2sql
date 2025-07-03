from peft import LoraConfig, get_peft_model, PeftModel
from transformers import PreTrainedModel
from modelscope import AutoModelForCausalLM
from config import TrainingConfig
import torch

class CoTModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__(base_model.config)
        self.model = base_model
        self.cot_head = torch.nn.Linear(
            self.model.config.hidden_size, 
            self.model.config.vocab_size,
            bias=False
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        lm_logits = outputs.logits
        cot_logits = self.cot_head(hidden_states)

        if labels is not None:
            # Convert labels to tensor if necessary
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long, device=lm_logits.device)
            # Using attention_mask to determine padding positions
            pad_mask = (attention_mask == 0).unsqueeze(-1)
            pad_mask = pad_mask.expand_as(lm_logits)
            final_logits = torch.where(pad_mask, lm_logits, cot_logits)
        else:
            final_logits = cot_logits

        return {"logits": final_logits}

def load_base_model():
    config = TrainingConfig()
    return AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 确保模型使用 float16 精度
    )

def setup_model(model):
    config = TrainingConfig()
    # LoRA配置
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type=config.task_type,
        target_modules=config.target_modules
    )
    
    return get_peft_model(model, peft_config)

# 加载已经训练好的LoRA模型
def load_lora(model_path):
    config = TrainingConfig()
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 确保模型使用 float16 精度
        device_map = 'auto'
    )
    
    # 加载LoRA适配器权重
    lora_model = PeftModel.from_pretrained(model, model_path, torch_dtype=torch.float16)
    
    # for param_name, param in lora_model.named_parameters():
    #     print(f"Parameter name: {param_name}, dtype: {param.dtype}")
    
    return lora_model
