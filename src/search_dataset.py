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

def calculate_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    intersection = max(0, inter_end - inter_start)
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return intersection / union


def evaluate_data(output_path: str, dataset_path: str, k=5, iou_threshold: float = 0.05):
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        student_search_results = data.get("search_results", [])

        dataset_results = parse_data_set(dataset_path)
        dataset_by_id = {item["question_id"]: item for item in dataset_results}

        per_question_recalls = []

        for student_result in student_search_results:
            qid = student_result["question_id"]
            dataset_result = dataset_by_id.get(qid)
            if dataset_result is None:
                print(f"Question ID {qid} not found in dataset; skipping.")
                continue

            if student_result["question"] != dataset_result["question"]:
                print(f"Question text mismatch for {qid}.")

            gold_sources = dataset_result["sources"]
            if not gold_sources:
                continue

            retrieved_sources = student_result["retrieved_sources"][:k]

            found = 0
            for gold in gold_sources:
                match = any(
                    retrieved["file_path"] == gold["file_path"]
                    and calculate_iou(
                        gold["first_character_index"], gold["last_character_index"],
                        retrieved["first_character_index"], retrieved["last_character_index"],
                    ) >= iou_threshold
                    for retrieved in retrieved_sources
                )
                if match:
                    found += 1

            per_question_recalls.append(found / len(gold_sources))

        if not per_question_recalls:
            print("No questions evaluated.")
            return 0.0, 0

        recall_at_k = sum(per_question_recalls) / len(per_question_recalls)
        print(f"Evaluation complete! Recall@{k}: {recall_at_k:.4f} over {len(per_question_recalls)} questions.")
        return recall_at_k, len(per_question_recalls)

    except Exception as e:
        print(f"Error during evaluation: {e} at line: {e.__traceback__.tb_lineno}")
        exit(99)


def main():
    evaluate_data("data/output/search_results/AnsweredQuestions/dataset_docs_public.json",
                  "data/datasets/AnsweredQuestions/dataset_docs_public.json")

if __name__ == "__main__":
    main()