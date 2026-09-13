import json

from llm_call import call_llm_foreach_query, generate_response, json_dump_search_and_answers, json_dump_search_results
from search_dataset import retrieve_question_id, retrieve_questions
from indexing import index_files, chromadb_indexing, get_metadata
from build_retrieved_data import build_retrieved_data, cached_retrieval, total_search_results
import fire, time

class Rag:
    def __init__(self):
        pass
    
    def index(self, max_chunk_size=2000):
        start_time = time.time()
        index_files(max_chunk_size)
        # chromadb_indexing(max_chunk_size)
        end_time = time.time()
        print(f"Ingestion complete! Indices saved under data/processed/ (Time taken: {end_time - start_time:.2f} seconds)")
    
    def search(self, query, k=5):
        #check if processed data exists, if not, run index
        data = build_retrieved_data(query, k)
        for record in data.retrieved_sources:
            print(f"File Path: {record.file_path}", end=" ")
            print(f"[{record.first_character_index}:{record.last_character_index}]")
    
    
    def search_dataset(self, dataset_path, k=5, save_directory="data/output/search_results"):
        question_ids = retrieve_question_id(dataset_path)
        questions = retrieve_questions(dataset_path)
        student_search_results = total_search_results(questions, question_ids, k)
        output_file = json_dump_search_results(
            student_search_results,
            save_directory + "/" + dataset_path.split("/")[-2] + "/" + dataset_path.split("/")[-1]
        )
        print(f"Saved student_search_results to {output_file}")

    
    def answer(self, query, k=5):
        metadata = self.index()
        minimal_search_results = build_retrieved_data(query, k)
        print(generate_response(minimal_search_results))
    
    def answer_dataset(self, student_search_results_path, save_directory):
        # data_file = self.search_dataset(dataset_path, k)
        data_file = json.load(open(student_search_results_path, "r"))
        queries = [item["question"] for item in data_file]
        student_result_and_answers = call_llm_foreach_query(data_file)
        output_file = json_dump_search_and_answers(student_result_and_answers,
                                 "data/output/search_results_and_answer/" + dataset_path.split("/")[-2] + "/" + dataset_path.split("/")[-1]
                                )
        return output_file


def main():
    # try:
    fire.Fire(Rag)
    # except Exception as e:
        # print(f"Error: {e}")

if __name__ == "__main__":
    main()
