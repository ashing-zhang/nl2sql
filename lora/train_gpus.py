'''
    0.cd ./lora
    1.使用deepspeed多卡训练
    2.训练命令：deepspeed train_gpus.py --deepspeed --deepspeed_config ds_config.json
    3.注意：train_batch_size = micro_batch_per_gpu * gradient_acc_step * world_size
'''
import os
import torch
import numpy as np
import json
import random
import deepspeed
from tqdm.auto import tqdm
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from modelscope import AutoModelForCausalLM
from model_setup import setup_model,load_lora
from data_loader import get_train_dataloader
from config import TrainingConfig
import argparse
from itertools import cycle

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

def train_process(args):
    set_seed(2025)
    deepspeed.init_distributed()
    
    # 加载DeepSpeed配置
    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)
    '''
        常规任务建议1000−3000步，具体取决于数据集规模：
        小数据集（<10万样本）：800−1500步
        中等数据集（100万样本）：2000−5000步
        大数据集（>500万样本）：5000−10000步
    '''
    total_steps = ds_config["scheduler"]["params"]["total_num_steps"]
    
    config = TrainingConfig()
    # 初始化基础模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 确保模型使用 float16 精度
    )
    model = setup_model(model)
    # 加载已经lora微调得到的适配器权重
    # model = load_lora(config.best_model)
    
    '''
        DeepSpeed初始化(args中包括了命令行输入的deepspeed_config，
        所以initialize中不能给config赋值，即不能使用config=)
    '''
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    # 分布式数据加载
    dataset = get_train_dataloader().dataset
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
        num_workers=4,
        pin_memory=True
    )
    data_iter = cycle(dataloader)  # 无限数据迭代

    # 训练循环
    global_step = 0
    best_loss = float('inf')
    progress_bar = tqdm(total=total_steps, desc="Training") if model_engine.local_rank == 0 else None
    
    while global_step < total_steps:
        model_engine.train()
        batch = next(data_iter)
        
        # 前向传播
        inputs = {k: v.to(model_engine.device) for k, v in batch.items()}
        outputs = model_engine(**inputs)
        loss = outputs.loss
        
        # 反向传播
        model_engine.backward(loss)
        model_engine.step()
        global_step += 1

        # 主进程更新进度
        if model_engine.local_rank == 0:
            progress_bar.update(1)
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # 早停判断逻辑（仅主进程执行）
        if model_engine.local_rank == 0 and global_step % 50 == 0:
            current_loss = loss.item()
            
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
    train_process(args)
    