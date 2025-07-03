1.项目概况

> 基于大语言模型LLM构建一个问答系统，问答内容涉及基金/股票/债券/招股书等不同数据来源

2.技术路线
> （1）data query
>
> - 大模型的幻觉：deepseek-ai/DeepSeek-R1-Distill-Qwen-7B直接进行nl2sql任务产生的结果较差，如产生幻觉、无法完成多表联合查询等（幻觉举例："question": "查询下基金代码008295的基金，它的管理费率是？","sql": "SELECT management_fees FROM fund_basic_info WHERE fund_code = '008295';"。但其实无表名fund_basic_info，也无字段名management_fees）。将数据库各表的元信息输入给deepseek R1，借助deepseek R1生成50条nl2sql数据，然后Lora微调deepseek-ai/DeepSeek-R1-Distill-Qwen-7B模型，以增强deepseek-ai/DeepSeek-R1-Distill-Qwen-7B对于nl2sql任务的能力。
> - 数据库的精准匹配：由于sql查询语句需要与数据库有非常精准的匹配（列名的精准匹配、表中数据格式的匹配等），因此deepseek R1生成的sql查询语句不可全信。告诉deepseek R1数据库中各表的结构，deepseek R1可以生成语法及逻辑正确但不符合表中数据格式的nl-sql语句对（原因应该是未告诉deepseek R1各个表中各列数据的具体格式），因此**后续在微调deepseek-ai/DeepSeek-R1-Distill-Qwen-7B后并使用它进行推理时，可以考虑在prompt中加入限定表中各列数据具体格式的内容**。
> - 微调数据的验证：需要检查构造的sql语句是否能从数据库返回正确的结果，因此需要执行生成的sql语句，检验构造数据的可行性。
> - 算力需求：RTX 4090 * 1卡足够lora微调（target_modules=["q_proj", "v_proj"]）deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
> - cot(chain of thought)数据构造：在构造nl-sql语句对的基础上增加思维链数据，使用该构造数据进行lora微调。在显存为32GB的显卡上对DeepSeek-R1-Distill-Qwen-7B进行lora微调，当max_length=512时，batch_size须不大于1才能不至于显存溢出。（使用50条cot数据对模型进行lora微调，效果并不好，nl2sql推理结果甚至比不上微调前的模型）

