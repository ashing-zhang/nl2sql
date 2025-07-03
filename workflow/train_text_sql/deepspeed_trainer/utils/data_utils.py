"""
Data utilities for DeepSpeed QLoRA training.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Union
import numpy as np
from datasets import Dataset, DatasetDict
import torch
import transformers

logger = logging.getLogger(__name__)

def preprocess(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message:str
) -> Dict:
    IGNORE_TOKEN_ID = -100

    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    # print('tokenizer.eod_id:', tokenizer.eod_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id
    # 如果tokenizer含有im_start_id,则im_start = tokenizer.im_start_id，否则im_start = tokenizer("<|im_start|>")["input_ids"][0]
    if hasattr(tokenizer, "im_start_id") and tokenizer.im_start_id is not None:
        im_start = [tokenizer.im_start_id]
    else:
        # Qwen2.5-7B-Instruct模型的tokenizer没有im_start_id
        im_start = tokenizer("<|im_start|>").input_ids
    if hasattr(tokenizer, "im_end_id") and tokenizer.im_end_id is not None:
        im_end = [tokenizer.im_end_id]
    else:
        im_end = tokenizer("<|im_end|>").input_ids
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    if roles[source[0]["from"]] != roles["user"]:
        source = source[1:]

    input_id, target = [], []
    system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
    input_id += system
    # 除了[im_start]、[im_end]和[NL]，其他token都要mask掉
    target += im_start + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end + nl_tokens
    assert len(input_id) == len(target)
    # 遍历每一条数据中的每个字典
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        # print('role:', role)
        # print('sentence["value"]:', sentence["value"])
        _input_id = tokenizer(role).input_ids + nl_tokens + \
            tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
        input_id += _input_id
        if role == '<|im_start|>user':
            _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
        elif role == '<|im_start|>assistant':
            '''
                1.im_start:这是助手的对话开始标记。通常，
                    我们希望模型学会生成这个标记，所以它在 _input_id 中。
                    但在 _target 中，这里也将其设为 im_start，
                    意味着我们希望模型学习在正确的时机预测这个开始标记。
                2.[IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids):代表了assistant和
                    nl_tokens的长度和
                3.im_end + nl_tokens：学会在合适的位置生成回合结束标记 <|im_end|> 和换行符 
            '''
            _target = im_start + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                _input_id[len(tokenizer(role).input_ids)+1:-2] + im_end + nl_tokens
        else:
            raise NotImplementedError
        target += _target
    assert len(input_id) == len(target)
    # print('tokenizer:', tokenizer)
    input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
    # print('tokenizer.pad_token_id:', tokenizer.pad_token_id)
    # print('input_id:', input_id)
    target += [IGNORE_TOKEN_ID] * (max_len - len(target))
    input_ids.append(input_id[:max_len])
    targets.append(target[:max_len])
    # print('input_ids:', input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id)
    )

def load_data(config, mode):
    if mode == 'train':
        data_path = os.path.join(config['data']['train_data_dir'],config['data']['train_json_path'])
    elif mode == 'val':
        data_path = os.path.join(config['data']['val_data_dir'],config['data']['val_json_path'])
    return json.load(open(data_path, "r"))

class TextSqlDataset(Dataset):
    def __init__(self, config, mode, tokenizer):
        self.config = config
        self.raw_data = load_data(config, mode)
        self.tokenizer = tokenizer
        self.processed_data = []
        for item in self.raw_data:
            one_data = preprocess(
                item['conversations'],
                self.tokenizer,
                self.config['data']['max_length'],
                system_message=self.config['data']['format']['system_message']
            )
            # 批量预处理所有数据
            self.processed_data.append(one_data)
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if isinstance(i, list):
            # 如果i是一个列表，返回对应索引的批量数据
            return self.__getitems__(i)
        item = self.processed_data[i]
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "labels": item["labels"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0)
        }
    
    def __getitems__(self, indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        batch = [self.processed_data[idx] for idx in indices]
        return [{
            "input_ids": item["input_ids"].squeeze(0),
            "labels": item["labels"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0)
        } for item in batch]
    
# return DatasetDict
def load_dataset(
    data_file: str,
    validation_file: Optional[str] = None,
    validation_split: float = 0.1,
    format_type: str = "instruction"
) -> DatasetDict:
    """加载数据集"""
    logger.info(f"Loading dataset from {data_file}")
    
    # open是被@contextmanager装饰器装饰的生成器函数
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f) 
    if format_type == "conversations":
        # 直接使用conversations格式的数据
        train_dataset = Dataset.from_list(data)
    
    if validation_file:
        with open(validation_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        if format_type == "conversations":
            validation_dataset = Dataset.from_list(val_data)  
    else:
        # 分割训练集和验证集
        if format_type == "conversations":
            train_size = int((1 - validation_split) * len(data))
            train_data = data[:train_size]
            val_data = data[train_size:]
            train_dataset = Dataset.from_list(train_data)
            validation_dataset = Dataset.from_list(val_data)
        
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })
    
    logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(validation_dataset)} validation")
    return dataset_dict

def process_conversations_data(
    dataset: Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int = 512,
    system_message: str = "You are a helpful assistant that can generate SQL queries based on natural language questions.",
    truncation: bool = True,
    padding: str = "max_length",
    return_tensors: str = "pt"
) -> Dataset:
    """处理conversations格式的数据"""
    logger.info("Preprocessing conversations dataset")
    
    IGNORE_TOKEN_ID = -100
    
    # 检查tokenizer是否有必要的特殊token
    if not hasattr(tokenizer, 'im_start_id') or not hasattr(tokenizer, 'im_end_id'):
        logger.warning("Tokenizer missing im_start_id or im_end_id, using fallback method")
        return process_conversations_fallback(dataset, tokenizer, max_length, system_message, truncation, padding, return_tensors)
    
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id
    
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens
    
    def tokenize_conversation(example):
        conversations = example['conversations']
        
        # 确保第一个对话是用户
        if conversations[0]["from"] != "user":
            conversations = conversations[1:]
        
        input_id, target = [], []
        
        # 添加系统消息
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        
        # 处理对话
        for sentence in conversations:
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError(f"Unknown role: {role}")
            
            target += _target
        
        # 填充到最大长度
        input_id += [tokenizer.pad_token_id] * (max_length - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_length - len(target))
        
        return {
            'input_ids': input_id[:max_length],
            'labels': target[:max_length],
            'attention_mask': [1 if id != tokenizer.pad_token_id else 0 for id in input_id[:max_length]]
        }
    
    # remove_columns=dataset.column_names 的作用是移除原始数据集中的所有列
    # 因为tokenize_conversation函数返回的是新的tokenized数据（input_ids, labels, attention_mask）
    # 原始数据中的conversations列已经不再需要，所以需要被移除
    tokenized_dataset = dataset.map(
        tokenize_conversation,
        remove_columns=dataset.column_names
    )
    
    logger.info("Conversations dataset preprocessing completed")
    return tokenized_dataset


def process_conversations_fallback(
    dataset: Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int = 512,
    system_message: str = "You are a helpful assistant that can generate SQL queries based on natural language questions.",
    truncation: bool = True,
    padding: str = "max_length",
    return_tensors: str = "pt"
) -> Dataset:
    """conversations格式的fallback处理方法"""
    logger.info("Using fallback method for conversations preprocessing")
    
    def format_conversation(example):
        conversations = example['conversations']
        formatted_text = f"System: {system_message}\n\n"
        
        for conv in conversations:
            role = conv["from"]
            value = conv["value"]
            formatted_text += f"{role.capitalize()}: {value}\n"
        
        return {'text': formatted_text}
    
    # 先格式化为文本
    formatted_dataset = dataset.map(format_conversation)
    
    # 然后使用标准的tokenization
    return preprocess_data(formatted_dataset, tokenizer, max_length, truncation, padding, return_tensors)


def preprocess_data(
    dataset: Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int = 512,
    truncation: bool = True,
    padding: str = "max_length",
    return_tensors: str = "pt"
) -> Dataset:
    """预处理数据集"""
    logger.info("Preprocessing dataset")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=truncation,
            pad_to_multiple_of=8,
            max_length=max_length,
            return_tensors=return_tensors
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info("Dataset preprocessing completed")
    return tokenized_dataset


def create_data_collator(tokenizer: transformers.PreTrainedTokenizer, padding: bool = True):
    """创建数据整理器"""
    from transformers import DataCollatorForLanguageModeling
    
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )


def get_dataset_info(dataset: Dataset) -> Dict:
    """获取数据集信息"""
    info = {
        'num_examples': len(dataset),
        'column_names': dataset.column_names,
        'features': dataset.features
    }
    
    if 'text' in dataset.column_names:
        text_lengths = [len(str(text)) for text in dataset['text']]
        info['text_length_stats'] = {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths),
            'median': np.median(text_lengths)
        }
    elif 'conversations' in dataset.column_names:
        # 计算conversations格式的数据统计信息
        conversation_lengths = [len(conv['conversations']) for conv in dataset['conversations']]
        total_messages = sum(conversation_lengths)
        info['conversation_stats'] = {
            'total_conversations': len(dataset),
            'total_messages': total_messages,
            'avg_messages_per_conversation': total_messages / len(dataset),
            'conversation_length_stats': {
                'mean': np.mean(conversation_lengths),
                'std': np.std(conversation_lengths),
                'min': np.min(conversation_lengths),
                'max': np.max(conversation_lengths),
                'median': np.median(conversation_lengths)
            }
        }
    
    return info


def validate_dataset(dataset: Dataset, required_columns: List[str]) -> bool:
    """验证数据集格式"""
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    for col in required_columns:
        if any(pd.isna(dataset[col])):
            logger.warning(f"Column {col} contains null values")
    
    return True
