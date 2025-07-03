# DeepSpeed QLoRA Training

这是一个使用 DeepSpeed 进行分布式 QLoRA (Quantized Low-Rank Adaptation) 训练的项目。

## 项目特性

- 🚀 使用 DeepSpeed ZeRO-3 进行分布式训练
- 🔧 支持 QLoRA 量化训练，大幅降低显存需求
- 📊 支持多GPU训练，提高训练效率
- 🎯 支持多种模型架构 (LLaMA, Mistral, Qwen等)
- 📈 完整的训练监控和日志记录
- 🛠️ 易于配置和扩展

## 项目结构

```
deepspeed_train/
├── configs/                 # 配置文件
│   ├── deepspeed_config.json
│   └── train_config.yaml
├── data/                    # 数据目录
│   └── sample_data.json
├── models/                  # 模型相关
│   └── __init__.py
├── scripts/                 # 训练脚本
│   ├── train.py
│   └── inference.py
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── data_utils.py
│   └── model_utils.py
├── requirements.txt
├── setup.py
└── README.md
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据

将您的训练数据放在 `data/` 目录下，格式为 JSON 文件：

```json
[
    {
        "instruction": "请解释什么是机器学习",
        "input": "",
        "output": "机器学习是人工智能的一个分支..."
    }
]
```

### 2. 配置训练参数

编辑 `configs/train_config.yaml` 文件：

```yaml
model:
  base_model: "microsoft/DialoGPT-medium"
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.1

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  max_length: 512

data:
  train_file: "data/sample_data.json"
  validation_split: 0.1
```

### 3. 开始训练

```bash
# 单GPU训练
python scripts/train.py

# 多GPU训练
deepspeed --num_gpus=2 scripts/train.py --deepspeed configs/deepspeed_config.json
```

## 配置说明

### DeepSpeed 配置

`configs/deepspeed_config.json` 包含 DeepSpeed 的配置：

- **ZeRO-3**: 启用 ZeRO-3 优化
- **Gradient Accumulation**: 支持梯度累积
- **Mixed Precision**: 使用混合精度训练
- **Offload**: 支持 CPU 和 NVMe 卸载

### 训练配置

`configs/train_config.yaml` 包含模型和训练参数：

- **模型配置**: 基础模型、LoRA 参数
- **训练配置**: 批次大小、学习率、训练轮数
- **数据配置**: 数据文件路径、验证集比例

## 高级用法

### 自定义模型

在 `models/` 目录下添加您的自定义模型：

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def create_model(base_model_name, lora_config):
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    peft_config = LoraConfig(**lora_config)
    model = get_peft_model(model, peft_config)
    return model
```

### 自定义数据集

在 `utils/data_utils.py` 中添加您的数据处理逻辑：

```python
def load_custom_dataset(file_path):
    # 实现您的数据加载逻辑
    pass
```

## 性能优化

### 显存优化

- 使用 QLoRA 量化，可减少 70% 显存使用
- 启用 DeepSpeed ZeRO-3，支持大模型训练
- 使用梯度检查点 (gradient checkpointing)

### 训练加速

- 混合精度训练 (FP16/BF16)
- 梯度累积
- 多GPU 并行训练

## 监控和日志

训练过程中会生成以下日志：

- **训练损失**: 实时显示训练和验证损失
- **学习率**: 学习率调度曲线
- **显存使用**: GPU 显存使用情况
- **训练速度**: 每秒处理的样本数

## 故障排除

### 常见问题

1. **显存不足**: 减小 batch_size 或启用更多卸载选项
2. **训练速度慢**: 检查是否启用了混合精度训练
3. **收敛问题**: 调整学习率和 LoRA 参数

### 调试模式

```bash
# 启用详细日志
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python scripts/train.py --debug
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License 