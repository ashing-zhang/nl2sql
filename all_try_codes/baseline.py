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
        
        # 初始化数据库连接
        self.conn = sqlite3.connect(config["db_path"])
        self.db_schema = self._parse_database_schema()
        
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

    def _parse_database_schema(self):
        schema = {}
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
            schema[table] = columns
        
        with open("db_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=4)
            
        return schema

    def _generate_sql_prompt(self, question):
        schema_str = "\n".join(
            ["表 {} ({})".format(
                    table, ', '.join(["{} {}".format(col['name'], col['type']) for col in cols])
                ) for table, cols in self.db_schema.items()]
        )

        return f"""### 数据库结构:
                {schema_str}

                ### 问题:
                {question}

                ### SQL查询:
                """

    def _validate_sql(self, sql):
        sql = sql.lower().strip()
        if not re.match(r"^\s*select", sql):
            return False
        forbidden = ["insert", "update", "delete", "drop", "alter", "create"]
        return not any(keyword in sql for keyword in forbidden)

    def _execute_sql(self, sql):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"[SQL Error] {str(e)}")
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
        prompts = [
            f"""
            判断问题类型：
            - 如果问题需要查询具体数值数据，回复data_query
            - 如果问题需要分析文本内容，回复text_comprehension
            - 其他类型回复other

            问题：{question}
            类型：
            """ for question in questions
        ]
        
        outputs = self.text_gen_pipe(
            prompts,
            max_new_tokens=20,
            stopping_criteria=stop_criteria_list,
            do_sample=False
        )
        
        return [output["generated_text"].strip().lower() for output in outputs]

    def structured_query(self, questions):
        sql_prompts = [self._generate_sql_prompt(question) for question in questions]
        inputs = self.sql_tokenizer(sql_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        generated = self.sql_model.generate(
            inputs["input_ids"],
            max_new_tokens=300,
            num_return_sequences=1,
            early_stopping=True
        )
        sql_queries = [self.sql_tokenizer.decode(gen, skip_special_tokens=True).split("### SQL查询:")[-1].split(";")[0] + ";" for gen in generated]
        
        results = []
        for sql in sql_queries:
            if self._validate_sql(sql):
                result = self._execute_sql(sql)
                results.append(result)
            else:
                results.append(None)
        
        return results

    def rag_answer(self, questions):
        contexts = []
        for question in questions:
            docs = self.vector_store.similarity_search(question, k=3)
            context = "\n".join([d.page_content for d in docs])
            contexts.append(context)
        
        prompts = [
            f"""基于以下信息回答问题：
            {context}

            问题：{question}
            答案：""" for context, question in zip(contexts, questions)
        ]
        
        responses = self.text_gen_pipe(
            prompts,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            stopping_criteria=stop_criteria_list
        )
        
        answers = [response["generated_text"].split("答案：")[-1].strip() for response in responses]
        return answers

    def process_batch_questions(self, batch):
        qids = [q_dict["id"] for q_dict in batch]
        questions = [q_dict["question"] for q_dict in batch]
        
        task_types = self.classify_task(questions)
        results = []
        
        for qid, question, task_type in zip(qids, questions, task_types):
            try:
                if "data_query" in task_type:
                    result = self.structured_query([question])[0]
                    if result:
                        answer = " ".join([str(r) for r in result])
                    else:
                        answer = self.rag_answer([question])[0]
                else:
                    answer = self.rag_answer([question])[0]
                    
            except Exception as e:
                answer = f"处理过程中发生错误：{str(e)}"
                
            results.append({
                "id": qid,
                "question": question,
                "answer": answer
            })
        
        return results

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
    config = {
        "text_gen_model": "models/Qwen2.5-7B-Instruct",
        "text2sql_model": "models/Qwen2.5-7B-Instruct",
        "embed_model": "models/all-MiniLM-L6-v2",
        "db_path": "data/dataset/fund_data.db",
        "txt_dir": "data/pdf_txt_file",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    }
    
    qa_system = FinancialQA(config)
    
    data_path = "data/question_debug.json"
    dataset = QuestionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(qa_system.process_batch_questions, batch) for batch in dataloader]
        for future in tqdm(futures, desc="Processing batches"):
            results.extend(future.result())
    
    with open("submit_result.jsonl", "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    print(f"Processed {len(results)} questions.")



