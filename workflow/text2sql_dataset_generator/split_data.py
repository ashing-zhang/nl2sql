'''
    python -m workflow.text2sql_dataset_generator.split_data
'''
import json
import random
from sklearn.model_selection import train_test_split

def split_text2sql_dataset(input_file, train_output_file, val_output_file, train_ratio=0.8):
    # 设置随机种子保证可复现性
    random.seed(42)
    
    # 读取原始数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 打乱数据顺序避免顺序偏差
    random.shuffle(dataset)
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(
        dataset,
        test_size=1 - train_ratio,
        shuffle=True
    )
    
    
    # 保存训练集
    with open(train_output_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # 保存验证集
    with open(val_output_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print(f"总数据量: {len(dataset)}")  
    print(f"训练集: {len(train_data)} 条 ({len(train_data)/len(dataset):.1%})")
    print(f"验证集: {len(val_data)} 条 ({len(val_data)/len(dataset):.1%})")

if __name__ == "__main__":
    # 输入文件路径
    source_file = "workflow/text2sql_dataset_generator/merged_data.json"
    train_output_file = "workflow/text2sql_dataset_generator/train_text_sql.json"
    val_output_file = "workflow/text2sql_dataset_generator/val_text_sql.json"

    split_text2sql_dataset(source_file,train_output_file,val_output_file)