import json
import sqlite3
import re
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    pipeline,
    StoppingCriteriaList
)
from modelscope import AutoTokenizer, AutoModelForCausalLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from stop_criteria_utils import MaxLengthStopCriteria, RepetitionStopCriteria, LowConfidenceStopCriteria
from concurrent.futures import ThreadPoolExecutor

# 组合多个终止条件
stop_criteria_list = StoppingCriteriaList([
    MaxLengthStopCriteria(200),  # 最长 200 个 token
    RepetitionStopCriteria(3),   # 避免 3 连重复
    LowConfidenceStopCriteria(0.1)  # 最高置信度低于 0.1 时停止
])

class FinancialQA:
    def __init__(self, config):
        # 初始化设备配置
        self.device = config["device"]
        self.is_cuda = "cuda" in self.device
        self.config = config
        
        # 初始化数据库连接
        self.conn = sqlite3.connect(config["db_path"])
        self.db_schema = json.load(open(config["db_schema_path"], "r", encoding="utf-8"))
        self.all_columns = [col["name"] for table in self.db_schema.values() for col in table]
        with open(config["db_keywords_path"], "r", encoding="utf-8") as f:
            self.db_keywords = [line.strip() for line in f if line.strip()]
        # 生成表结构描述
        self.table_descs = []
        for table_name, columns in self.db_schema.items():
            col_details = "\n".join([f"    ▪ {col['name']} ({col['type']})" for col in columns])
            self.table_descs.append(f"▌ 表名：{table_name}\n{col_details}")
        # 从rules.txt文件中加载转换规则
        with open('rules.txt', 'r', encoding='utf-8') as f:
            self.rules = f.read()
        
        # 从data/example_queries.txt文件中加载业务特定示例
        self.example_queries = []
        with open(self.config["example_queries_path"], 'r', encoding='utf-8') as f:
            for line in f:
                question, sql = line.strip().split(", ", 1)
                self.example_queries.append(('自然语言查询：'+ question, '对应的sql查询：' + sql))
        print('self.example_queries[0]:',self.example_queries[0])
        # 初始化文本生成模型
        self.text_gen_tokenizer = AutoTokenizer.from_pretrained(
            config["text_gen_model"], 
            trust_remote_code=True
        )
        self.text_gen_model = AutoModelForCausalLM.from_pretrained(
            config["text_gen_model"],
            torch_dtype=torch.float16 if self.is_cuda else torch.float32,
        ).to(self.device)
        
        self.text_gen_pipe = pipeline(
            "text-generation",
            model=self.text_gen_model,
            tokenizer=self.text_gen_tokenizer,
            device=self.device
        )
        
        # 初始化Text-to-SQL模型（与文本生成模型相同）
        self.sql_tokenizer = self.text_gen_tokenizer
        self.sql_model = self.text_gen_model

        # 初始化RAG系统
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["embed_model"],
            model_kwargs={"device": self.device.split(":")[0], "local_files_only": False},
            encode_kwargs={"normalize_embeddings": False}
        )
        persist_dir = "data/chroma_db"+"/"+config["embed_model"]
        self.vector_store = self._init_vector_store(config["txt_dir"], persist_dir)
    

    def _generate_sql_prompt(self, question):
        return f"""
            📌 任务描述：
            你是一名 SQL 生成专家，需要将用户的自然语言问题转换为 **高效且准确的 SQL 查询**。
            
            📊 **数据库模式（Schema）**：
            {'\n\n'.join(self.table_descs)}

            📝 **SQL 生成要求：**
            {self.rules}

            🛠️ 根据以上信息（包括数据库模式、SQL生成要求等），
            请将以下问题转换为SQL语句：
            {question}
            🎯 **最终 SQL 查询**：
            
            """.strip()

    def _sanitize_sql(self, sql):
        sql = sql.strip().replace("```sql", "").replace("```", "")  # 清理标记
        # print('sql:', sql)
        
        # 找出sql中的所有字段
        # sql_columns = []
        # for column in self.all_columns:
        #     if column in sql:
        #         sql_columns.append(column)
        
        # print('sql after sanitize:', sql)
        return sql

    def _execute_sql(self, question, sql):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            
            # Combine question and result to generate a natural language answer
            prompt = f"问题：{question}\nSQL查询结果：{result}\n请根据查询结果生成自然语言答案（仅给出答案即可，无需重复问题）："
            inputs = self.text_gen_tokenizer(prompt, return_tensors="pt").to(self.device)
            generated = self.text_gen_model.generate(
                inputs["input_ids"],
                max_new_tokens=150,
                early_stopping=True
            )
            answer = self.text_gen_tokenizer.decode(generated[0], skip_special_tokens=True)
            return answer.strip()
        except sqlite3.Error as e:
            print(f"[SQL Error] {str(e)}")
            # 报错时存储问题、SQL和错误信息
            with open('sql_log.json', 'a', encoding='utf-8') as log_file:
                json.dump(
                    {"question": question, "sql": sql, "error": str(e)},
                    log_file,
                    ensure_ascii=False,
                    indent=4
                )
                log_file.write("\n")
            return None

    def _init_vector_store(self, txt_dir, persist_dir):
        if os.path.exists(persist_dir):
            print(f"向量数据库已存在于 {persist_dir}，跳过初始化")
            return Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        
        docs = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        for filename in tqdm(os.listdir(txt_dir), desc="加载文档"):
            if filename.endswith(".txt"):
                with open(os.path.join(txt_dir, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                docs.extend(splitter.create_documents([text]))
        
        return Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )

    def classify_task(self, questions):
        # 去除标点符号，从而不干扰question_vectors的生成，进而不影响相似度计算
        questions = ["".join(c for c in q if c not in "?!？！，。") for q in questions]
        def determine_task_type(question):
            # 确保编码正确
            question = question.encode("utf-8").decode("utf-8")

            # 使用 encode() + decode() 方式进行分词
            token_ids = self.text_gen_tokenizer.encode(question, add_special_tokens=False)
            # question_words = set(self.text_gen_tokenizer.decode([token]) for token in token_ids)
            '''
            分词结果示例：
                question_words: {'年', '基金', '嘉', '?', '实', '基金管理', '有限公司', '0', '2', '成立了', '9', '多少', '1'}
                question_words: {'技术', '股份', '负责', '生物', '有限公司', '？', '产品研发', '森', '沃', '云南', '什么', '部门', '的是'}
            '''
            # print("question_words:", question_words)

            match_count = sum(1 for char in question if any(char in keyword for keyword in self.db_keywords))
            if match_count / len(question) >= 0.3:
                return "data_query"
            else:
                docs = self.vector_store.similarity_search(question, k=1)
                return "text_comprehension" if docs else "other"

        return [determine_task_type(question) for question in questions]

    def structured_query(self, question):
        sql_prompt = self._generate_sql_prompt(question)
        # print('question:', question)
        # print('sql_prompt:', sql_prompt)
        inputs = self.sql_tokenizer(sql_prompt, return_tensors="pt").to(self.device)
        
        generated = self.sql_model.generate(
            inputs["input_ids"],
            max_new_tokens=500,
            num_return_sequences=3,
            early_stopping=True
        )
        sql = self.sql_tokenizer.decode(generated[0], skip_special_tokens=True)
        sql = sql.split("```sql\n")[-1].split(";")[0] + ";"
        # print('sql:', sql)
        
        # sql = self._sanitize_sql(sql)
        
        # Store the question and corresponding SQL in a file
        with open('question_to_sql.json', 'a', encoding='utf-8') as f:
            json.dump({"question": question, "sql": sql}, f, ensure_ascii=False, indent=4)
            f.write("\n")
        
        return self._execute_sql(question,sql)
        
    def rag_answer(self, questions):
        contexts = []
        for question in questions:
            docs = self.vector_store.similarity_search(question, k=3)
            context = "\n".join([d.page_content for d in docs])
            contexts.append(context)

        prompts = [f"""基于以下信息回答问题：
                    {context}

                    问题：{question}
                    答案：""" for context, question in zip(contexts, questions)]
        
        responses = self.text_gen_pipe(prompts, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2, stopping_criteria=stop_criteria_list)

        return [response[0]["generated_text"].split("答案：")[-1].strip() for response in responses]

    def process_batch(self, qid_batch, question_batch):
        task_types = self.classify_task(question_batch)
        answers = []
        
        with open('question_type.json', 'a', encoding='utf-8') as f:
            for task_type, question in zip(task_types, question_batch):
                json.dump({"question": question, "task_type": task_type}, f, ensure_ascii=False, indent=4)
                f.write("\n")
                answer = "No valid answer found"  # Default initialization
                if task_type == "data_query":
                    result = self.structured_query(question)
                    # print('question:', question)
                    # print('result:', result)
                    if result:
                        answer = " ".join([str(r) for r in result])
                else:
                    answer = self.rag_answer([question])[0]
                
                answers.append(answer)
        
        return [{"id": qid, "question": question, "answer": answer} for qid, question, answer in zip(qid_batch, question_batch, answers)]


class QuestionDataset(Dataset):
    def __init__(self, data_path):
        self.questions = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.questions.append(json.loads(line))

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx]

# 使用示例
if __name__ == "__main__":
    # 清除question_to_sql.json中的内容
    with open('question_to_sql.json', 'w', encoding='utf-8') as f:
        f.write("")
    # 清除sql_log.json中的内容
    with open('sql_log.json', 'w', encoding='utf-8') as f:
        f.write("")
    # 清除question_type.json中的内容
    with open('question_type.json', 'w', encoding='utf-8') as f:
        f.write("")
    
    config = {
        # "text_gen_model": "models/Qwen2.5-7B-Instruct",
        # "text2sql_model": "models/Qwen2.5-7B-Instruct",
        "text_gen_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "text2sql_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "embed_model": "models/all-MiniLM-L6-v2",
        "db_path": "data/dataset/fund_data.db",
        "db_schema_path": "data/dataset/db_schema.json",
        "example_queries_path": "data/example_queries.txt",
        "txt_dir": "data/pdf_txt_file",
        "db_keywords_path": "data/dataset/db_keywords.txt",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    }
    
    qa_system = FinancialQA(config)
    
    data_path = "data/question_debug.json"
    # data_path = "data/question.json"
    dataset = QuestionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

    # 使用批量推理
    results = []
    for batch in tqdm(dataloader, desc="Processing questions"):
        qid_batch = batch['id'].tolist()
        question_batch = batch['question']
        results.extend(qa_system.process_batch(qid_batch, question_batch))

    # Ensure all tensors are converted to lists before writing to JSON
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, list):
            return [tensor_to_list(item) for item in obj]
        if isinstance(obj, dict):
            return {key: tensor_to_list(value) for key, value in obj.items()}
        return obj

    results = tensor_to_list(results)
    # print('results:', results)
    with open("submit_result.jsonl", "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    print(f"Processed {len(results)} questions.")
