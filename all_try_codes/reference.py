import json
import sqlite3
import re
import os
from tqdm import tqdm
import torch
from transformers import (
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
from modelscope import AutoTokenizer, AutoModelForCausalLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from stop_criteria_utils import MaxLengthStopCriteria,RepetitionStopCriteria,LowConfidenceStopCriteria

# 组合多个终止条件
stop_criteria_list = StoppingCriteriaList([
    MaxLengthStopCriteria(200),  # 最长 200 个 token
    RepetitionStopCriteria(3),   # 避免 3 连重复
    LowConfidenceStopCriteria(0.1)  # 最高置信度低于 0.1 时停止
])

class FinancialQA:
    def __init__(self, config):
        """
        config = {
            "text_gen_model": "Qwen/Qwen2.5-7B",  # 文本生成模型
            "text2sql_model": "defog/sqlcoder-7b-2", # Text-to-SQL专用模型
            "embed_model": "shibing624/text2vec-base-chinese",
            "db_path": "data/dataset/fund_data.db",
            "txt_dir": "data/pdf_txt_file",
            "device": "cuda:0"  # or "cpu"
        }
        """
        # 初始化设备配置
        self.device = config["device"]
        self.is_cuda = "cuda" in self.device
        
        # 初始化数据库连接
        self.conn = sqlite3.connect(config["db_path"])
        # 数据库中各数据表的元数据
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
        
        # 初始化Text-to-SQL模型
        self.sql_tokenizer = AutoTokenizer.from_pretrained(config["text2sql_model"])
        self.sql_model = AutoModelForCausalLM.from_pretrained(
            config["text2sql_model"],
            torch_dtype=torch.float16 if self.is_cuda else torch.float32,
        ).to(self.device)

        # print('config["embed_model"]:',config["embed_model"])
        # 初始化RAG系统
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["embed_model"],
            model_kwargs={"device": self.device.split(":")[0], "local_files_only": False},
            encode_kwargs={"normalize_embeddings": False}
        )
        persist_dir = "data/chroma_db"+"/"+config["embed_model"]
        self.vector_store = self._init_vector_store(config["txt_dir"],persist_dir)

    def _parse_database_schema(self):
        """动态解析数据库结构并存储到文件"""
        schema = {}
        cursor = self.conn.cursor()
        
        # 获取所有表信息
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # 获取每个表的列信息
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
            schema[table] = columns
        
        # 将schema存储到文件
        with open("db_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=4)
            
        return schema

    def _generate_sql_prompt(self, question):
        """生成Text-to-SQL的提示模板"""
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
        """SQL语句安全验证"""
        sql = sql.lower().strip()
        # 检查是否SELECT语句
        if not re.match(r"^\s*select", sql):
            return False
        # 禁止危险操作
        forbidden = ["insert", "update", "delete", "drop", "alter", "create"]
        return not any(keyword in sql for keyword in forbidden)

    def _execute_sql(self, sql):
        """执行SQL并返回结果"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"[SQL Error] {str(e)}")
            return None

    def _init_vector_store(self, txt_dir, persist_dir):
        """初始化向量数据库，如果已经存在则跳过"""
        if os.path.exists(persist_dir):
            print(f"向量数据库已存在于 {persist_dir}，跳过初始化")
            # 需要 embedding 参数并不是为了重新计算已存向量，而是为了确保新数据和查询都使用一致的嵌入模型
            return Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        
        docs = []
        '''
            递归地按字符将文本分割成更小的块，以便于后续处理
        '''
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, # 每个块的最大字符数
            chunk_overlap=100 # 相邻块之间的重叠字符数，确保信息在块之间的连续性
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

    def classify_task(self, question):
        """问题分类"""
        prompt = f"""
                    判断问题类型：
                    - 如果问题需要查询具体数值数据，回复data_query
                    - 如果问题需要分析文本内容，回复text_comprehension
                    - 其他类型回复other

                    问题：{question}
                    类型：
                """
        
        output = self.text_gen_pipe(
            prompt,
            max_new_tokens=20,
            stopping_criteria=stop_criteria_list,
            do_sample=False
        )[0]["generated_text"]
        
        return output.strip().lower()

    def structured_query(self, question):
        """结构化数据查询"""
        # 生成SQL
        sql_prompt = self._generate_sql_prompt(question)
        # print("sql_prompt:",sql_prompt)
        inputs = self.sql_tokenizer(sql_prompt, return_tensors="pt").to(self.device)
        print('inputs:',inputs)
        
        generated = self.sql_model.generate(
            inputs["input_ids"],
            max_new_tokens=300,
            num_return_sequences=1,
            early_stopping=True
        )
        sql = self.sql_tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # 清理输出（去除提示部分）
        sql = sql.split("### SQL查询:")[-1].split(";")[0] + ";"
        
        # 验证并执行
        if not self._validate_sql(sql):
            return None
        
        return self._execute_sql(sql)

    def rag_answer(self, question):
        """检索增强生成"""
        docs = self.vector_store.similarity_search(question, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        prompt = f"""基于以下信息回答问题：
                {context}

                问题：{question}
                答案："""
        
        response = self.text_gen_pipe(
            prompt,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            stopping_criteria=stop_criteria_list
        )
        return response[0]["generated_text"].split("答案：")[-1].strip()

    def process_question(self, qid, question):
        """处理完整流程"""
        try:
            task_type = self.classify_task(question)
            
            if "data_query" in task_type:
                result = self.structured_query(question)
                if result:
                    answer = " ".join([str(r) for r in result])
                else:
                    answer = self.rag_answer(question)
            else:
                answer = self.rag_answer(question)
                
        except Exception as e:
            answer = f"处理过程中发生错误：{str(e)}"
            
        return {
            "id": qid,
            "question": question,
            "answer": answer
        }

# 使用示例
if __name__ == "__main__":
    config = {
        "text_gen_model": "models/Qwen2.5-7B-Instruct",
        "text2sql_model": "models/sqlcoder-7b-2",
        # "embed_model": 可以用模型"BGE（BAAI General Embedding）",效果可能更好
        # "embed_model": "models/baai",
        # "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embed_model":"models/all-MiniLM-L6-v2",
        "db_path": "data/dataset/fund_data.db",
        "txt_dir": "data/pdf_txt_file", # 招股说明书所在的folder
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    }
    
    qa_system = FinancialQA(config)
    
    # 从 question.json 文件中读取问题列表
    test_questions = []
    with open("data/question_debug.json", "r", encoding="utf-8") as f:
        for line in f:
            test_questions.append(json.loads(line.strip()))

    results = []
    for q in tqdm(test_questions, desc="Processing questions"):
        result = qa_system.process_question(q["id"], q["question"])
        results.append(result)
    
    # 将结果写入 JSONL 文件
    with open("submit_result.jsonl", "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")