import json
import os

from llm_call import call_llm_foreach_query, generate_response, json_dump_search_and_answers, json_dump_search_results
from search_dataset import evaluate_data, retrieve_question_id, retrieve_questions
from indexing import index_files, chromadb_indexing
from build_retrieved_data import build_retrieved_data, total_search_results, total_chromadb_search_results, reciprocal_rank_fusion
from data_models import StudentSearchResults, MinimalSearchResults, MinimalSource
from pathlib import Path
import fire, time

class Rag:
    def __init__(self):
        pass
    
    def index(self, max_chunk_size=2000):
        start_time = time.time()
        index_files(max_chunk_size)
        chromadb_indexing(max_chunk_size)
        end_time = time.time()
        print(f"Ingestion complete! Indices saved under data/processed/ (Time taken: {end_time - start_time:.2f} seconds)")
    
    def search(self, query, k=5):
        #check if processed data exists, if not, run index
        if not os.path.exists("data/processed/"):
            self.index()
        bm25_results = total_search_results([query], [''], k)
        chroma_res = total_chromadb_search_results([query], [''], k)
    
        fused = reciprocal_rank_fusion(bm25_results, chroma_res)
        for record in fused[0][:k]:
            print(f"File Path: {record.file_path}", end=" ")
            print(f"[{record.first_character_index}:{record.last_character_index}]")
    
    

    def search_dataset(self, dataset_path, k=10, save_directory="data/output/search_results"):
        if save_directory != "data/output/search_results":
            output_path = save_directory
        elif "data/datasets/AnsweredQuestions/" in dataset_path or "data/datasets/UnansweredQuestions/" in dataset_path:
            output_path = save_directory + "/" + dataset_path.split("/")[-2] + "/" + dataset_path.split("/")[-1]
        else:
            output_path = "data/output/search_results/user_output.json"
            
        if not os.path.exists(output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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

    
    def answer(self, query, k=5):
        # metadata = self.index()
        if not os.path.exists("data/processed/"):
                            self.index()
        bm25_results = total_search_results([query], [''], k)
        chroma_res = total_chromadb_search_results([query], [''], k)
        fused = reciprocal_rank_fusion(bm25_results, chroma_res)
        minimal_search_results = MinimalSearchResults(
            question_id='',
            question=query,
            retrieved_sources=fused[0][:k]
        )
        print(generate_response(minimal_search_results))
    
    def answer_dataset(self, student_search_results_path, save_directory="data/output/search_results_and_answers"):
        if save_directory != "data/output/search_results_and_answers":
            output_path = save_directory
        elif ("data/datasets/AnsweredQuestions" or "data/datasets/UnansweredQuestions") in student_search_results_path and save_directory == "data/output/search_results_and_answers":
            output_path = save_directory + "/" + student_search_results_path.split("/")[-2] + "/" + student_search_results_path.split("/")[-1]
        elif ("data/datasets/AnsweredQuestions" or "data/datasets/UnansweredQuestions") not in student_search_results_path:
            output_path = "data/output/search_results_and_answers/user_output.json"
        if not os.path.exists(output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(student_search_results_path, 'r') as f:
            student_search_results = json.load(f)
        responses = call_llm_foreach_query(StudentSearchResults(**student_search_results))
        answers = json_dump_search_and_answers(
            StudentSearchResults(**student_search_results),
            responses,
            output_path
        )
        print(f"Saved student_search_results_and_answers to {answers}")
    
    def evaluate(self, student_search_results_path, dataset_path):
        evaluate_data(student_search_results_path, dataset_path)

def main():
    # try:
    fire.Fire(Rag)
    # except Exception as e:
        # print(f"Error: {e}")

if __name__ == "__main__":
    main()
