import json

def remove_id_fields(input_file, output_file):
    """
    移除JSON数据中所有条目的"id"字段
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出JSON文件路径
    """
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建新数据集
    cleaned_data = []
    for item in data:
        # 创建条目副本并删除id字段
        new_item = item.copy()
        if "id" in new_item:
            del new_item["id"]
        cleaned_data.append(new_item)
    
    # 写入处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_path = "val_text_sql.json"
    output_path = "val_text_sql.json"
    remove_id_fields(input_path, output_path)
    print(f"处理完成，已生成新文件：{output_path}")