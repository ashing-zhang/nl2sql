'''
    从modelscope下载模型
'''
from modelscope import snashot_download

model_dir = snashot_download("DeepSeek-R1-Distill-Qwen-7B")
print('模型下载完成，模型路径为：', model_dir)