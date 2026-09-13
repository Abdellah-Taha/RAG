from typing import List
from build_retrieved_data import total_search_results, total_chromadb_search_results
import json

from data_models import MinimalSource

path_code = "data/datasets/AnsweredQuestions/dataset_code_public.json"
path_docs = "data/datasets/AnsweredQuestions/dataset_docs_public.json"

def parse_data_set(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        son = json.load(f)
    
    return son.get("rag_questions", [])

def retrieve_questions(file_path: str):
    data_set = parse_data_set(file_path)
    return [item["question"] for item in data_set]

def retrieve_data_source(file_path: str):
    data_set = parse_data_set(file_path)
    return [item["sources"] for item in data_set]

def retrieve_question_id(file_path: str):
    data_set = parse_data_set(file_path)
    return [item["question_id"] for item in data_set]
