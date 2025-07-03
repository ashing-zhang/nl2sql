from db.db_connection import query_db
from prompts.load_prompt import load_template
from prompts.prompt_templates import sql_gen_prompt,answer_summary_prompt
from apis.model_api import llm

def handle_sql_query(question: str) -> str:
    prompt_sql = load_template(sql_gen_prompt,question)
    sql = llm.generate(prompt_sql)
    print('sql:', sql)
    # llm生成的sql语句成了随后工具调用的输入
    result = query_db(sql)
    prompt_answer = load_template(answer_summary_prompt,question,result)
    # llm将sql语句的执行结果进行总结
    result = llm.generate(prompt_answer)
    return result
