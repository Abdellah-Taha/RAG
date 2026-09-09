from functools import lru_cache
from typing import List
from data_models import StudentSearchResults, MinimalSearchResults, MinimalSource,StudentSearchResultsAndAnswer, MinimalAnswer
from transformers import AutoModelForCausalLM, AutoTokenizer
import re, json


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class Llm:
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens=150):
        message = [
            {"role": "system", "content": (
                "You are a helpful assistant. Answer the question using only the "
                "contextual information provided. Give a direct, concise answer — "
                "state the fact directly without restating the question or explaining "
                "your reasoning. If the context includes a specific endpoint, command, "
                "or value, quote it exactly."
            )},
            {"role": "user", "content": prompt}
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        input_text = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True)

        input_length = input_text["input_ids"].shape[1]

        output = self.model.generate(**input_text,
                                    max_new_tokens=max_new_tokens,
                                    use_cache=True,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id)

        generated_tokens = output[0][input_length:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return strip_thinking(output_text)

@lru_cache()
def call_llm():
    return Llm()

def extract_text_from_context(context: MinimalSearchResults):
    for path in context.retrieved_sources:
        with open(path.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            yield text[path.first_character_index:path.last_character_index]

def generate_response(context: MinimalSearchResults, max_new_tokens: int = 150):
    llm = call_llm()
    result: List[str] = []
    for text in extract_text_from_context(context):
        result.append(text)
    super_prompt = f"{context.question}\n\nContext:\n" + "\n".join(result) + "\nResponse:"
    return str(llm.generate(super_prompt, max_new_tokens))

def call_llm_foreach_query(context: StudentSearchResults):
    responses = []
    for i, result in enumerate(context.search_results):
        response_text = generate_response(result)
        responses.append(response_text)
        
        print(f"Question: {result.question}")
        print(f"response: {responses[i]}")
        print("=======================================================")
    return responses

def create_student_search_results_and_answer(context: StudentSearchResults, responses: List[str]) -> StudentSearchResultsAndAnswer:
    minimal_answers: List[MinimalAnswer] = []
    for i, result in enumerate(context.search_results):
        minimal_answer = MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=responses[i]
        )
        minimal_answers.append(minimal_answer)

    return StudentSearchResultsAndAnswer(
        search_results=minimal_answers,
        k=context.k
    )

def json_dump_search_results(answers: StudentSearchResultsAndAnswer, output_file: str):
    search_results_list = {
        "search_results": [],
        "k": answers.k
    }
    for data in answers.search_results:
        data_dict = {
            "question_id": data.question_id,
            "question": data.question,
            "retrieved_sources": [
                {
                    "file_path": source.file_path,
                    "first_character_index": source.first_character_index,
                    "last_character_index": source.last_character_index
                }
                for source in data.retrieved_sources
            ]
        }
        search_results_list["search_results"].append(data_dict)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(search_results_list, f, ensure_ascii=False, indent=4)
        
def json_dump_search_and_answers(answers: StudentSearchResultsAndAnswer, output_file: str):
    search_results_list = {
        "search_results": [],
        "k": answers.k
    }
    for data in answers.search_results:
        data_dict = {
            "question_id": data.question_id,
            "question": data.question,
            "retrieved_sources": [
                {
                    "file_path": source.file_path,
                    "first_character_index": source.first_character_index,
                    "last_character_index": source.last_character_index
                }
                for source in data.retrieved_sources
            ],
            "answer": data.answer
        }
        search_results_list["search_results"].append(data_dict)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(search_results_list, f, ensure_ascii=False, indent=4)
