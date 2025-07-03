'''
    (1)Lora微调deepseek-ai/DeepSeek-R1-Distill-Qwen-7B模型，
        以增强deepseek-ai/DeepSeek-R1-Distill-Qwen-7B对于nl2sql任务的能力
    (2)利用DeepSeek-R1生成微调所用的数据：采用优先生成SQL、再生成问题的数据生成顺序，有利于提高SQL生成的准确性
        (DeepSeek-R1生成的nl2sql数据并不能直接使用，需要人工对比数据库对生成数据进行校验)。
'''
