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



def evaluate_data(output_path:str, dataset_path:str, k=5):
    try:
        correct, total = 0, 0
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        student_search_results = data.get("search_results", [])
        questions = retrieve_questions(dataset_path)
        data_sources = retrieve_data_source(dataset_path)
        for i, student_result in enumerate(student_search_results):
            question = student_result.get("question", "")
            retrieved_sources = student_result.get("retrieved_sources", [])
            correct_sources = data_sources[i]
            total += 1
            if len(retrieved_sources) == 0:
                print(f"Question: {question} - No sources retrieved.")
                continue
            # Check if any of the retrieved sources match the correct sources
            if any(source.get("file_path") in [correct_source.get("file_path") for correct_source in correct_sources] for source in retrieved_sources):
                correct += 1
                print(f"Question: {question} - Correct source found.")
            else:
                print(f"Question: {question} - No correct source found.")
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Evaluation complete! Accuracy: {accuracy:.2f}% ({correct}/{total})")
    except Exception as e:
        print(f"Error during evaluation: {e} at line : {e.__traceback__.tb_lineno}")
        exit(99)
    
def main():
    evaluate_data("data/output/search_results/AnsweredQuestions/dataset_docs_public.json",
                  "data/datasets/AnsweredQuestions/dataset_docs_public.json")

if __name__ == "__main__":
    main()