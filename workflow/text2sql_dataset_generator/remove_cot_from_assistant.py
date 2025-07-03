'''
    python -m workflow.text2sql_dataset_generator.remove_cot_from_assistant
'''
import json
import re

input_path = 'workflow/text2sql_dataset_generator/text2sql_dataset_add.json'
output_path = 'workflow/text2sql_dataset_generator/text2sql_dataset_add.json'

# 匹配sql:（不区分大小写，中英文冒号）及其后内容
sql_pattern = re.compile(r'(sql:|SQL:|sql：|SQL：)(.*)', re.DOTALL)

def remove_cot_from_value(value):
    match = sql_pattern.search(value)
    if match:
        return match.group(2).lstrip()  # 只保留sql:之后的内容
    return ''  # 没有sql:则返回空字符串

def process_file():
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified = False
    for item in data:
        if 'conversations' in item:
            for conv in item['conversations']:
                if conv.get('from') == 'assistant' and isinstance(conv.get('value'), str):
                    new_value = remove_cot_from_value(conv['value'])
                    if new_value != conv['value']:
                        conv['value'] = new_value
                        modified = True

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"处理完成，结果已保存到 {output_path}。{'有修改' if modified else '无cot内容被删除'}")

if __name__ == '__main__':
    process_file() 