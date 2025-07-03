from retriever.doc_retrieve import retrieve
from prompts.load_prompt import load_template
from prompts.prompt_templates import answer_summary_prompt
from apis.model_api import llm


def handle_doc_query(question: str) -> str:
    content = retrieve(question)
    prompt = load_template(answer_summary_prompt,question,content)
    result = llm.generate(prompt)
    return result
