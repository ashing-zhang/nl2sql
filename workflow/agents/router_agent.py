from prompts.load_prompt import load_template
from prompts.prompt_templates import route_prompt
from apis.model_api import llm

def classify_query(question: str) -> str:
    prompt = load_template(route_prompt,question)
    result = llm.generate(prompt)
    return "doc" if "doc" in result else "sql"
