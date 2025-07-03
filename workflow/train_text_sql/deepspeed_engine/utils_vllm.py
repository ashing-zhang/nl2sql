from vllm import SamplingParams
# 多轮对话生成函数
def qwen_chat_vllm(llm, tokenizer, messages, max_tokens=512):
    # 1. 转换对话格式
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    # 2. 配置生成参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_tokens,
        extra_params={"repetition_penalty": 1.05, "top_k": 20}
    )
    # 3. 生成文本
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text