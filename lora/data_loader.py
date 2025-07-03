import os
from torch.utils.data import Dataset, DataLoader
from modelscope import AutoTokenizer
from config import TrainingConfig

class SQLDataset(Dataset):
    def __init__(self, tokenizer, mode='train'):
        self.tokenizer = tokenizer
        self.mode = mode  # 新增mode参数，'train' 或 'inference'
        self.config = TrainingConfig()
        
        # 验证数据完整性
        self._validate_data()
        self.samples = self._load_samples()

    def _validate_data(self):
        if self.mode == 'train':
            q_dir = os.path.join(self.config.data_root, self.config.question_dir)
            l_dir = os.path.join(self.config.data_root, self.config.label_dir)
            if not os.path.exists(q_dir) or not os.path.exists(l_dir):
                raise FileNotFoundError("请检查数据目录结构是否符合要求")
        else:
            self.predict_file_path = os.path.join(self.config.data_root, self.config.predict_file)
            if not os.path.exists(self.predict_file_path):
                raise FileNotFoundError("推理模式下问题数据缺失")

    def _load_samples(self):
        samples = []
        if self.mode == 'train':
            base_path = os.path.join(self.config.data_root, self.config.question_dir)
            for i in range(1, 51):
                q_path = os.path.join(base_path, f"{i}.txt")
                l_path = os.path.join(self.config.data_root, 
                                      self.config.label_dir, 
                                      f"{i}.txt")
                
                with open(q_path, "r", encoding="utf-8") as f:
                    question = f.read().strip()
                    
                with open(l_path, "r", encoding="utf-8") as f:
                    sql = f.read().strip()
                
                samples.append((question, sql))
        elif self.mode == 'inference':
            with open(self.predict_file_path, "r", encoding="utf-8") as f:
                questions = f.readlines()
            for question in questions:
                samples.append((question.strip(), None))  # 只有问题，不需要SQL标签
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question, sql = self.samples[idx]
        return self._tokenize(question, sql)

    def _tokenize(self, question, sql):
        if self.mode == 'train':
            input_text = f"Question: {question}\nSQL: "
            target_text = sql
        else:  # 推理模式
            input_text = f"Question: {question}\nSQL: "
            target_text = ""  # 无目标文本
        
        tokenized = self.tokenizer(
            input_text,
            text_target=target_text,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        if self.mode == 'train':
            return {
                "input_ids": tokenized["input_ids"].squeeze(),
                "attention_mask": tokenized["attention_mask"].squeeze(),
                "labels": tokenized["labels"].squeeze()
            }
        else:
            return {
                "input_ids": tokenized["input_ids"].squeeze(),
                "attention_mask": tokenized["attention_mask"].squeeze()
            }

def get_train_dataloader():
    tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.model_name)
    dataset = SQLDataset(tokenizer, mode='train')
    return DataLoader(
        dataset,
        batch_size=TrainingConfig.batch_size,
        shuffle=True
    )

def get_inference_dataloader():
    tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.model_name)
    dataset = SQLDataset(tokenizer, mode='inference')
    return DataLoader(
        dataset,
        batch_size=1,  # 推理时通常批量大小为1
        shuffle=False
    )
