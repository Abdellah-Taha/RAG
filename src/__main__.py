import json
import os

from .llm_call import (
    call_llm_foreach_query,
    generate_response,
    create_student_search_results_and_answer,
    json_dump_search_and_answers,
    json_dump_search_results,
)
from .search_dataset import (
    evaluate_data, retrieve_question_id, retrieve_questions)
from .indexing import (
    index_files, chromadb_indexing, files_mtime)
from .build_retrieved_data import (
    total_search_results,
    total_chromadb_search_results,
    reciprocal_rank_fusion,
)
from .data_models import (
    StudentSearchResults, MinimalSearchResults
    )
from pathlib import Path
import fire
import time
from typing import Any


class Rag:
    def __init__(self) -> None:
        pass

    def index(self, max_chunk_size: int = 2000) -> Any:
        start_time = time.time()
        index_files(max_chunk_size)
        chromadb_indexing(max_chunk_size)
        end_time = time.time()
        print(
            f"Ingestion complete! Indices saved under data/processed/\
(Time taken: {end_time - start_time:.2f} seconds)"
        )

    def search(self, query: str, k: int = 5) -> Any:
        data_path = Path("data/processed")
        data_raw = Path("data/raw")
        raw_files_date = files_mtime(data_raw)
        processed_files_date = files_mtime(data_path)
        if (raw_files_date > processed_files_date) or not data_path.exists():
            self.index()
        bm25_results = total_search_results([query], [""], k)
        chroma_res = total_chromadb_search_results([query], [""], k)

        fused = reciprocal_rank_fusion(bm25_results, chroma_res)
        for record in fused[0][:k]:
            print(f"File Path: {record.file_path}", end=" ")
            print(f"[{record.first_character_index}\
:{record.last_character_index}]")

    def search_dataset(self,
                       dataset_path: str,
                       k: int = 10,
                       save_directory: str = "data/output") -> Any:
        output_path = save_directory + "/" + Path(dataset_path).name

        question_ids = retrieve_question_id(dataset_path)
        questions = retrieve_questions(dataset_path)
        bm25_results = total_search_results(questions, question_ids, k)
        chroma_res = total_chromadb_search_results(questions, question_ids, k)
        fused = reciprocal_rank_fusion(bm25_results, chroma_res)
        output_file = json_dump_search_results(
            StudentSearchResults(
                search_results=[
                    MinimalSearchResults(
                        question_id=question_ids[i],
                        question=questions[i],
                        retrieved_sources=fused[i][:k],
                    )
                    for i in range(len(questions))
                ],
                k=k,
            ),
            output_path,
        )

        print(f"Saved student_search_results to {output_file}")

    def answer(self, query: str, k: int = 5) -> Any:
        if not os.path.exists("data/processed/"):
            self.index()
        if query.strip() == "":
            print("Query cannot be empty.")
            return
        bm25_results = total_search_results([query], [""], k)
        chroma_res = total_chromadb_search_results([query], [""], k)
        fused = reciprocal_rank_fusion(bm25_results, chroma_res)
        minimal_search_results = MinimalSearchResults(
            question_id="", question=query, retrieved_sources=fused[0][:k]
        )
        print(generate_response(minimal_search_results))

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str = "data/output"
                       ) -> Any:

        output_path = save_directory + "/\
" + Path(student_search_results_path).name

        with open(student_search_results_path, "r") as f:
            student_search_results = json.load(f)
        context = StudentSearchResults(**student_search_results)
        responses = call_llm_foreach_query(context)
        results_and_answers = create_student_search_results_and_answer(
            context, responses
        )
        answers = json_dump_search_and_answers(
            results_and_answers,
            output_path
        )
        print(f"Saved student_search_results_and_answers to {answers}")

    def evaluate(self,
                 student_search_results_path: str,
                 dataset_path: str,
                 k: int = 5,
                 ) -> Any:
        evaluate_data(student_search_results_path, dataset_path, k)


def main() -> None:
    try:
        fire.Fire(Rag)
    except BaseException as e:
        print(f"Error: {e}")
        exit(100)


if __name__ == "__main__":
    main()
