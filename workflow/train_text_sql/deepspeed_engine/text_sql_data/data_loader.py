import transformers
from typing import Dict, Optional, List
from torch.utils.data import Dataset
import torch
import os
import json

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message:str
) -> Dict:
    IGNORE_TOKEN_ID = -100

    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    # print('tokenizer.eod_id:', tokenizer.eod_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eod_id
    im_start = tokenizer("<|im_start|>")["input_ids"][0]
    im_end = tokenizer("<|im_end|>")["input_ids"][0]
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        # 除了[im_start]、[im_end]和[NL]，其他token都要mask掉
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        # 遍历每一条数据中的每个字典
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # print('role:', role)
            # print('sentence["value"]:', sentence["value"])
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                '''
                    1.[im_start]:这是助手的对话开始标记。通常，
                      我们希望模型学会生成这个标记，所以它在 _input_id 中。
                      但在 _target 中，这里也将其设为 im_start，
                      意味着我们希望模型学习在正确的时机预测这个开始标记。
                    2.[IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids):代表了assistant和
                      nl_tokens的长度和
                    3.[im_end] + nl_tokens：学会在合适的位置生成回合结束标记 <|im_end|> 和换行符 
                '''
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
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
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def load_data(config, mode):
    if mode == 'train':
        data_path = os.path.join(config.train_data_dir,config.train_json_path)
    elif mode == 'val':
        data_path = os.path.join(config.val_data_dir,config.val_json_path)
    return json.load(open(data_path, "r"))

class TextSqlDataset(Dataset):
    def __init__(self,config,mode):
        self.config = config
        self.raw_data = load_data(config,mode)
        # 禁用缓存，防止内存泄露
        # self.cached_data_dict = {}
        
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 禁用缓存，每次都重新处理
        # if i in self.cached_data_dict:
        #     return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.config.tokenizer, self.config.max_len,system_message = self.config.system_message)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        
        # 禁用缓存
        # self.cached_data_dict[i] = ret

        return ret

    
    