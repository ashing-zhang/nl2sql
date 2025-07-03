from model.llm_factory import LLMFactory

model_name = 'qwen'

llm = LLMFactory.build_llm(model_name,{})