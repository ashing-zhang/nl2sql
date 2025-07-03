'''
    cd ./workflow/train_text_sql/deepspeed_engine
    训练命令：deepspeed finetune.py --deepspeed --deepspeed_config ds_config.json
'''

import deepspeed
from transformers import AutoModel, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import Config
import random
import numpy as np
import os
import shutil
import torch
import argparse
from torch.utils.data import DataLoader
from text_sql_data.data_loader import TextSqlDataset
from torch.utils.data.distributed import DistributedSampler
from itertools import cycle
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def seed_everything(seed):
    # 添加分布式种子偏移
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    seed += local_rank  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 早停控制器
class EarlyStopper:
    def __init__(self, patience=25, improvement_ratio=0.9):
        self.patience = patience
        self.improvement_ratio = improvement_ratio  # 改进比例阈值（默认90%）
        self.counter = 0
        self.best_loss = float('inf')
    
    def check_improvement(self, val_loss):
        # 计算改进阈值
        improvement_threshold = self.best_loss * self.improvement_ratio
        
        if val_loss < improvement_threshold:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False
        
def add_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',type=int,default=-1)
    parser.add_argument('--debug',action='store_true',default=False)
    parser = deepspeed.add_config_arguments(parser)  # 自动添加 DeepSpeed 参数
    return parser.parse_args()

def validate(engine, val_loader):
    engine.eval()
    device = engine.device
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)  # 初始化张量
    total_samples = torch.tensor(0, device=device, dtype=torch.int32)

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            labels = inputs["labels"]
            
            outputs = engine(**inputs)
            loss = outputs.loss
            
            batch_size = torch.tensor(labels.shape[0], device=device)
            total_loss += loss.detach() * batch_size  # 保持张量运算
            total_samples += batch_size

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)
    
    avg_loss = total_loss.item() / (total_samples.item() + 1e-10)
    return avg_loss
            
            
def train(args):
    config = Config()
    seed_everything(config.seed)

    # 初始化分布式环境
    deepspeed.init_distributed()

    # 加载DeepSpeed配置
    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",         # 使用QLoRA设计的NF4量化
        bnb_4bit_use_double_quant=True,    # 启用双重量化压缩
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    train_micro_batch_size_per_gpu = ds_config["train_micro_batch_size_per_gpu"]
    gradient_accumulation_steps = ds_config["gradient_accumulation_steps"]

    local_rank = int(os.environ["LOCAL_RANK"])  # 通过环境变量获取当前进程号
    torch.cuda.set_device(local_rank)  # 显式绑定当前进程到指定GPU
    
    # 初始化模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        quantization_config=bnb_config,   # 当模型为Tongyi-Finance-14B-Chat-Int4，不宜再使用QLoRA量化
        torch_dtype=torch.bfloat16,         
        trust_remote_code=True
    )
    # 打印所有参数名称
    # for name, _ in model.named_parameters():
    #     print(name)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config.lora_config, adapter_name=config.adapter_name)
    # 打印model中可以更新的参数名的总数量
    trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    # Number of trainable parameter names before DeepSpeed engine wrapping: 392（单卡）
    print(f"Number of trainable parameter names before DeepSpeed engine wrapping: {len(trainable_param_names)}")
    # Model is on device: cuda:0（单卡时）
    # print(f"Model is on device: {next(model.parameters()).device}")

    # DeepSpeed初始化（分布式训练）
    # deepspeed.initialize -> DeepSpeedEngine.init()
    '''
        return_items = [
            engine,
            engine.optimizer,
            engine.training_dataloader,
            engine.lr_scheduler,
        ]
    '''
    # 每个 GPU（进程）上都会初始化一个 engine
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model
    )
    # 打印engine.module中每个参数的shape
    '''
        即便是单GPU，打印内容如下：
            base_model.model.model.embed_tokens.weight: (0,)
            base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight: (0,)
            base_model.model.model.layers.0.self_attn.q_proj.base_layer.bias: (0,)
            base_model.model.model.layers.0.self_attn.q_proj.lora_A.sql_lora.weight: (0,)
            base_model.model.model.layers.0.self_attn.q_proj.lora_B.sql_lora.weight: (0,)
            base_model.model.model.layers.0.self_attn.k_proj.base_layer.weight: (0,)
            base_model.model.model.layers.0.self_attn.k_proj.base_layer.bias: (0,)
            ......
        即模型经过deepspeed引擎包装后，其中的参数仅以占位符形式存在
    '''
    # for name, param in engine.module.named_parameters():
    #     print(f"{name}: {tuple(param.shape)}")
    
    train_dataset = TextSqlDataset(config,'train')
    val_dataset = TextSqlDataset(config,'val')
    # print("engine.local_rank:", engine.local_rank)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=engine.world_size,
        rank=engine.local_rank,
        shuffle=True,
        drop_last=True  # 添加此参数避免最后一个批次尺寸不一致
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=engine.world_size,
        rank=engine.local_rank,
        shuffle=True
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=config.tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # 提升GPU计算效率
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_micro_batch_size_per_gpu,
        sampler=train_sampler,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(config.seed + worker_id),
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_micro_batch_size_per_gpu,
        sampler=val_sampler,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(config.seed + worker_id),
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=collator
    )
    train_data_iter = cycle(train_loader)  # 无限数据迭代器
    if args.debug:
        total_steps = 50
    else:
        total_steps = config.epochs * len(train_loader)  # 计算总步数
    
    # 获取可训练参数名称列表
    trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    trainable_params_names = [name for name, _ in trainable_params]  # 提取参数名
    print('len(trainable_params_names):', len(trainable_params_names))
    # 写入文件
    with open('trainable_params.txt', 'w') as f:
        for param_name in trainable_params_names:
            f.write(f"{param_name}\n")  # 按行写入名称

    print(f"成功写入前 {len(trainable_params_names)} 个可训练参数名称到 trainable_params.txt")

    # 早停初始化
    early_stopper = EarlyStopper(
        patience=config.early_stop_patience,
        improvement_ratio = config.improvement_ratio
    )
    
    # 训练循环(每训练一个micro_batch_per_gpu * gradient_acc_step，model_engine.global_steps自增1)
    progress_bar = tqdm(total=total_steps, desc="Training Progress")

    train_losses = []
    val_losses = []

    cur_step = 0
    best_val_loss = float('inf')

    save_dir = os.path.join(config.sql_model_save_path,config.adapter_name)
    engine.train()
    # 将shape为0的参数个数追加写入到文件
    zero_shape_log_path = os.path.join(save_dir, "zero_shape_param_count.log")
    # 在进入while循环之前，确保zero_shape_log_path中无内容
    with open(zero_shape_log_path, 'w') as f:
        pass
    while cur_step < total_steps:
        # # 解包engine，获得module，打印其中每个tensor的shape
        # module = engine.module  # 解包engine获得底层模型
        # print("==== module tensors and their shapes ====")
        # '''
        #     适配器中lora相关权重的shape为0，其为占位符
        # '''
        # for name, param in module.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {tuple(param.shape)}")
        # print("==== end of module tensor shapes ====")

        batch = next(train_data_iter)

        # batch["lables"].device: cpu
        # print('batch["lables"].device:', batch["labels"].device)
        
        # 打印batch["input_features"]的数据类型
        # batch["input_features"]: <class 'torch.Tensor'>
        # print('batch["input_features"]:', type(batch["input_features"]))
        
        device = torch.device(f"cuda:{engine.local_rank}")  # 显式指定当前进程的GPU
        # device: cuda:0
        # print('device:', device)
        inputs = {
            key: tensor.to(device, non_blocking=True) if isinstance(tensor, torch.Tensor) else tensor
            for key, tensor in batch.items()
        }
        
        outputs = engine(**inputs)
        loss = outputs.loss
        # 反向传播（梯度处理）
        engine.backward(loss)
        engine.step()

        # 解包engine，获得module，打印其中每个tensor的shape，并记录shape为0的参数个数到文件
        os.system('clear')  # 清空terminal内容
        module = engine.module  # 解包engine获得底层模型
        # print("==== module tensors and their shapes ====")
        '''
            适配器中lora相关权重的shape为0，其为占位符
        '''
        zero_shape_count = 0
        for name, param in module.named_parameters():
            if param.requires_grad:
                # print(f"{name}: {tuple(param.shape)}")
                if param.numel() == 0 or any([s == 0 for s in param.shape]):
                    zero_shape_count += 1
        # print("==== end of module tensor shapes ====")
        
        with open(zero_shape_log_path, "a") as f:
            f.write(f"Step {cur_step}: zero-shape param count = {zero_shape_count}\n")

        # 清除梯度
        engine.zero_grad()
        
        # 更新进度条
        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())
        
        # 记录训练损失
        train_losses.append(loss.item())
        
        # 日志记录
        if cur_step % gradient_accumulation_steps == 0:
            print(f"Step {cur_step} | Loss: {loss.item():.4f}")
            # 验证阶段（分布式同步）
            # 验证阶段使用FP16精度
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                avg_val_loss = validate(engine, val_loader)
            val_losses.append(avg_val_loss)
            
            # 更新最佳验证损失和对应指标
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            # 早停和保存逻辑
            if early_stopper.check_improvement(avg_val_loss):
                # 如果save_dir存在且非空，则清空其内容
                if os.path.exists(save_dir) and os.listdir(save_dir):
                    for filename in os.listdir(save_dir):
                        file_path = os.path.join(save_dir, filename)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                engine.save_checkpoint(save_dir,exclude_frozen_parameters=True)
                print("Model and tokenizer saved with val loss: {:.4f}".format(avg_val_loss))
                
            else:
                if early_stopper.counter >= early_stopper.patience:
                    print("Early stopping triggered.")
                    break
        cur_step += 1

    progress_bar.close()

    # 绘制训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.legend()
    plt.savefig(config.train_loss_plot_path)
    print("Train loss plot saved as train_loss_plot.png")

    # 绘制验证损失
    plt.figure(figsize=(10, 5))
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()
    plt.savefig(config.val_loss_plot_path)
    print("Validation loss plot saved as val_loss_plot.png")

    # 打印最佳验证损失及对应指标
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    

if __name__ == "__main__":
    args = add_argument()
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用并行
    torch.cuda.empty_cache()
    train(args)