'''
    使用自己构造的cot(chain of thought)数据进行训练
    0.cd ./lora
    1.使用deepspeed多卡训练
    2.训练命令：deepspeed train_cot.py --deepspeed --deepspeed_config ds_config.json
    3.注意：train_batch_size = micro_batch_per_gpu * gradient_acc_step * world_size
'''
import os
import json
import random
import deepspeed
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data.distributed import DistributedSampler
from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedModel
from config import TrainingConfig
import argparse
from itertools import cycle
from model_setup import load_base_model, setup_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',type=int,default=-1)
    parser = deepspeed.add_config_arguments(parser)  # 自动添加 DeepSpeed 参数
    return parser.parse_args()

class CoTDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # 构建CoT输入格式：[问题] 逐步思考：
        prompt = f"{sample['question']}\n逐步思考："
        # 拼接CoT步骤和答案
        cot_steps = "\n".join([json.dumps(step) for step in sample['cot_steps']])
        full_answer = f"{cot_steps}\n最终答案：{sample['sql']}"
        
        # 编码输入和输出
        inputs = self.tokenizer(
            prompt, 
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        targets = self.tokenizer(
            full_answer,
            max_length=self.max_length,
            truncation=True, 
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze(),
        }

def train(args):
    set_seed(2025)
    deepspeed.init_distributed()

    # 加载DeepSpeed配置
    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)
    total_steps = ds_config["scheduler"]["params"]["total_num_steps"]
    
    # 加载配置
    config = TrainingConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    base_model = load_base_model()
    model = setup_model(base_model)
    
    # DeepSpeed初始化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )
    
    # 数据加载
    dataset = CoTDataset(config.full_cot_data_file, tokenizer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=model_engine.world_size,
        rank=model_engine.local_rank,
        shuffle=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4
    )
    data_iter = cycle(dataloader)  # 无限数据迭代
    
    # 训练循环
    global_step = 0
    best_loss = float('inf')
    progress_bar = tqdm(total=total_steps, disable=not model_engine.local_rank == 0)
    
    while global_step < total_steps:
        model_engine.train()
        for batch in dataloader:
            inputs = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model_engine(**inputs)
            
            # 计算CoT加权损失
            logits = outputs['logits']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            shift_mask = inputs["attention_mask"][..., 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            # 应用CoT损失权重
            weighted_loss = (loss * shift_mask.view(-1)).mean()
            
            # 反向传播
            model_engine.backward(weighted_loss)
            model_engine.step()
            global_step += 1
            
            # 进度更新
            if model_engine.local_rank == 0:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f"{weighted_loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
            # 早停判断逻辑（仅主进程执行）
        if model_engine.local_rank == 0 and global_step % 50 == 0:
            current_loss = weighted_loss.item()
            
            # 损失改进判断
            if (best_loss - current_loss) > config.min_delta:
                best_loss = current_loss
                no_improve_steps = 0
                model.save_pretrained(config.best_model,dtype=torch.float16)
                with open(f"{config.save_dir}/log.txt", "a") as log_file:
                    log_file.write(f"Best model saved at step {global_step} with loss {current_loss:.4f}\n")
            else:
                no_improve_steps += 1
                print(f"No improvement for {no_improve_steps} checks")
                
            # 触发早停条件
            if no_improve_steps >= config.early_stop_patience:
                print(f"Early stopping at step {global_step}")
                break  # 终止训练循环

if __name__ == "__main__":
    args = add_argument()
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    train(args)