'''
    一些工具方法
'''

# 将lora/data/train_question文件夹中的所有txt中的question放入一个列表中并打印
import os
import json
from modelscope import AutoTokenizer

def read_txt_files(folder_path):
    questions = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            idx = filename.split('.')[0]
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                questions.append((idx,content))
    return questions

# 长度分布验证脚本
def show_max_length(train_data_path):
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    lengths = []

    with open(train_data_path) as f:
        data = json.load(f)
        for sample in data:
            # 计算输出部分总长度
            cot_text = "\n".join([f"{s['step']}. {' '.join(s['content'])}" for s in sample['cot_steps']])
            full_answer = f"{cot_text}\n最终答案：{sample['sql']}"
            tokens = tokenizer(full_answer)["input_ids"]
            lengths.append(len(tokens))

    # 分析90%分位数
    import numpy as np
    # 90%样本长度 ≤ 300.1
    print(f"90%样本长度 ≤ {np.percentile(lengths, 90)}")

if __name__ == '__main__':
    # folder_path = './data/train_question'
    # questions_list = read_txt_files(folder_path)
    # for question in questions_list:
    #     print(str(question[0])+'.'+question[1])

    train_data_path = "data/train_cot_data/train.json"
    show_max_length(train_data_path)
