'''
    对data/question.json中的前100条问题进行分类，并将类别为'data_query'的问题输出到data/data_query.txt文件中
'''
import json
import os
def classify_query_task(questions, db_keywords_path, data_query_path):
    # 读取db_keywords
    with open(db_keywords_path, "r") as f:
        db_keywords = [line.strip() for line in f.readlines()]

    # 删除data_query_path中的内容
    if os.path.exists(data_query_path):
        os.remove(data_query_path)

    # 处理每个问题
    for question in questions:
        # 去除标点符号，从而不干扰question_vectors的生成，进而不影响相似度计算
        question = "".join(c for c in question if c not in "?!？！，。")
        
        # 确保编码正确
        question = question.encode("utf-8").decode("utf-8")

        match_count = sum(1 for char in question if any(char in keyword for keyword in db_keywords))
        if match_count / len(question) >= 0.6:
            with open(data_query_path, "a") as f:
                f.write(question + "\n")

if __name__ == "__main__":
    # 问题文件路径
    question_path = "data/question.json"
    # 读取前100条问题
    # questions = [json.loads(line)["question"] for line in open(question_path).readlines()][:100]
    questions = [json.loads(line)["question"] for line in open(question_path).readlines()]
    # "data_query"问题存储路径
    data_query_path = "data/data_query.txt"
    # db_keywords文件路径
    db_keywords_path = "data/dataset/db_keywords.txt"
    
    classify_query_task(questions, db_keywords_path, data_query_path)