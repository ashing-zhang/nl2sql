from utils.file_io import read_questions, write_answers
from agents.router_agent import classify_query
from agents.sql_agent import handle_sql_query
from agents.doc_agent import handle_doc_query
from tools.tool_dict import Tool

def main():
    # Initialize the tool dictionary
    tool_dict = Tool()
    # Add tools to the dictionary
    tool_dict.add_tool("sql", handle_sql_query)
    tool_dict.add_tool("doc", handle_doc_query)
    questions = read_questions("../data/question.json")
    answers = []
    for item in questions:
        qid, query = item["id"], item["question"]
        # llm的输出
        qtype = classify_query(query)
        print('question:', query)
        print('qtype:', qtype)
        # 输出的内容可以触发工具的调用
        answer = tool_dict.get_tool(qtype)(query)
        answers.append({"id": qid, "question": query, "answer": answer})
    write_answers("../data/answer.jsonl", answers)

if __name__ == '__main__':
    main()
