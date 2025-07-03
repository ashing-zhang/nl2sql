'''
    使用LLM（如deepseek）对自然语言进行解析，生成SQL语句，产生的nl-sql对作为微调数据来训练
    大模型（如Qwen7B等）适应nl2sql任务
    （直接调用接口处理带有自然语言查询的prompt，生成对应的SQL语句）
'''
