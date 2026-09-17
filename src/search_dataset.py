import json
from typing import Any

path_code = "data/datasets/AnsweredQuestions/dataset_code_public.json"
path_docs = "data/datasets/AnsweredQuestions/dataset_docs_public.json"


def parse_data_set(file_path: str) -> list[dict[str, Any]]:
    """Load the question records from a dataset JSON file.

    Args:
        file_path: Path to the dataset JSON file.

    Returns:
        Question records under the dataset's ``rag_questions`` key.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        son = json.load(f)
    result: list[dict[str, Any]] = son.get("rag_questions", [])
    return result


def retrieve_questions(file_path: str) -> Any:
    """Extract question text from a dataset file.

    Args:
        file_path: Path to the dataset JSON file.

    Returns:
        Question strings in dataset order.
    """
    data_set = parse_data_set(file_path)
    return [item["question"] for item in data_set]


def retrieve_data_source(file_path: str) -> Any:
    """Extract gold source references from a dataset file.

    Args:
        file_path: Path to the dataset JSON file.

    Returns:
        Source-reference collections in dataset order.
    """
    data_set = parse_data_set(file_path)
    return [item["sources"] for item in data_set]


def retrieve_question_id(file_path: str) -> Any:
    """Extract question identifiers from a dataset file.

    Args:
        file_path: Path to the dataset JSON file.

    Returns:
        Question identifiers in dataset order.
    """
    data_set = parse_data_set(file_path)
    return [item["question_id"] for item in data_set]


def calculate_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Calculate intersection over union for two character ranges.

    Args:
        a_start: Start offset of the first range.
        a_end: End offset of the first range.
        b_start: Start offset of the second range.
        b_end: End offset of the second range.

    Returns:
        The ratio of the intersection length to the union length.
    """
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    intersection = max(0, inter_end - inter_start)
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return intersection / union


def evaluate_data(
    output_path: str,
    dataset_path: str,
    k: int = 5,
    iou_threshold: float = 0.05
) -> Any:
    """Evaluate retrieved sources against the gold dataset.

    Args:
        output_path: Path to the student search-results JSON file.
        dataset_path: Path to the gold dataset JSON file.
        k: Number of retrieved sources to evaluate per question.
        iou_threshold: Minimum range overlap required for a source match.

    Returns:
        A tuple containing mean recall at ``k`` and the number evaluated.
    """
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

            if "sources" not in dataset_result:
                print(f"[{qid}] missing 'sources'. \
Available keys: {list(dataset_result.keys())}")
                continue
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
                        gold["first_character_index"],
                        gold["last_character_index"],
                        retrieved["first_character_index"],
                        retrieved["last_character_index"],
                    )
                    >= iou_threshold
                    for retrieved in retrieved_sources
                )
                if match:
                    found += 1

            per_question_recalls.append(found / len(gold_sources))

        if not per_question_recalls:
            print("No questions evaluated.")
            return 0.0, 0

        recall_at_k = sum(per_question_recalls) / len(per_question_recalls)
        print(
            f"Evaluation complete! Recall@{k}: \
{recall_at_k:.4f} over {len(per_question_recalls)} questions."
        )
        return recall_at_k, len(per_question_recalls)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        exit(99)
