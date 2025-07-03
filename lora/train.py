'''
    cd to lora
    deepspeed --num_gpus=3 train.py --deepspeed_config ds_config.json
'''
import os
import torch
import numpy as np
import random
import deepspeed
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from model_setup import setup_model
from data_loader import get_dataloader
from config import TrainingConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser = deepspeed.add_config_arguments(parser)  # 关键修改
    return parser.parse_args()

def train_process(args, rank, world_size):
    args = add_argument()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.set_device(rank)
    
    # 设置全局随机种子
    set_seed(2025)
    
    # DeepSpeed模型初始化
    model = setup_model()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # 初始化DeepSpeed引擎
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters,
        config=args.deepspeed_config
    )
    
    # 数据加载器配置
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)

    dataset = get_dataloader().dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(2025)
    )
    
    # 训练循环
    previous_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        if rank == 0:
            global_progress = tqdm(
                total=len(dataloader),
                desc=f"Epoch {epoch+1}/{config.num_epochs}",
                bar_format="{l_bar}{bar:20}{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            inputs = {k: v.to(model_engine.local_rank) for k, v in batch.items()}
            
            # DeepSpeed前向传播和反向传播
            outputs = model_engine(**inputs)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            
            # 同步loss
            avg_loss = loss.item()
            total_loss += avg_loss
            
            if rank == 0:
                global_progress.update(1)
                global_progress.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
        
        # 计算epoch平均loss
        epoch_loss = total_loss / len(dataloader)
        
        # 主进程保存模型
        if model_engine.local_rank == 0:
            if epoch_loss < previous_loss:
                save_path = f"{config.save_dir}/best_epoch_{epoch+1}"
                model_engine.save_checkpoint(save_path)
                print(f"\nLoss improved from {previous_loss:.4f} to {epoch_loss:.4f}, saved at {save_path}")
                previous_loss = epoch_loss
            
            if rank == 0:
                global_progress.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    world_size = 3  # GPU数量
    
    # 启动DeepSpeed训练
    deepspeed.launcher.launch(
        train_process,
        args=(args, args.local_rank, world_size),
        num_nodes=1,
        num_gpus=world_size
    )