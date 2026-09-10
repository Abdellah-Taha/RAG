from llm_call import call_llm_foreach_query, generate_response, json_dump_search_and_answers, json_dump_search_results
from search_dataset import retrieve_question_id, retrieve_questions
from indexing import index_files, chromadb_indexing
from build_retrieved_data import build_retrieved_data, cached_retrieval, total_search_results
import fire

# def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--k", type=int, default=5)
    # parser.add_argument("--max_chunk_size", type=int, default=2000)
    # parser.add_argument("--dataset_path", type=str, default="RAG/data/datasets/AnsweredQuestions/dataset_docs_public.json")
    # parser.add_argument("--max_chunk_size", type=int, )
    # args = parser.parse_args()
    # output_search_file = "data/output/search_results/" + args.dataset_path.split("/")[-2] + "/" + args.dataset_path.split("/")[-1]
    # output_search_and_answer_file = "data/output/search_results_and_answer/" + args.dataset_path.split("/")[-2] + "/" + args.dataset_path.split("/")[-1]
    # print(output_search_file)
    # print(output_search_and_answer_file)
    # start_time = time.time()
    # indexing the data files
    # meta_data = index_files(args.max_chunk_size)
    # meta_data = chromadb_indexing(args.max_chunk_size)
    # end_time = time.time()
    # print(f"Indexing complete in {end_time - start_time:.2f} seconds. You can now use the BM25 retriever for searching.")
    # retrieving the questions from the json file
    # questions = retrieve_questions(args.dataset_path)
    # question_ids = retrieve_question_id(args.dataset_path)
    # retrieving the relevent data for each query (RA ANA LI KANTB HACHI MACHI AI AW9S) 
    # search_results = total_search_results(questions, question_ids, args.k, meta_data=meta_data) # bm25 search results
    # search_results = total_chromadb_search_results(questions, question_ids, args.k, meta_data=meta_data) # chromadb search results
    # sending the search results to the llm
    # start = time.time()
    # student_result_and_answers = call_llm_foreach_query(search_results)
    # end = time.time()
    # print(f"\nTime taken: {end - start:.2f}")
    # json_dump_search_and_answers(student_result_and_answers, output_search_and_answer_file)
    # json_dump_search_results(student_result_and_answers, output_search_file)

class Rag:
    def __init__(self):
        pass
    
    def index(self, max_chunk_size=2000):
        metadata = index_files(max_chunk_size)
        print("Ingestion complete! Indices saved under data/processed/")
        return metadata
    
    def search(self, query, k=5):
        metadata = self.index() 
        
        results = cached_retrieval(query, k)
        
        for doc_idx in results[0][0]:
            idx = int(doc_idx)
            print(f"File Path: {metadata[idx]['file_path']}", end=" ")
            print(f"[{metadata[idx]['start']}:{metadata[idx]['end']}]")
            
        return results
    
    def search_dataset(self, dataset_path, k=5):
        metadata = self.index()
        question_ids = retrieve_question_id(dataset_path)
        questions = retrieve_questions(dataset_path)
        student_search_results = total_search_results(questions, question_ids, k, meta_data=metadata)
        output_file = json_dump_search_results(student_search_results,
                                 "data/output/search_results/" + dataset_path.split("/")[-2] + "/" + dataset_path.split("/")[-1]
                                )
        print(f"Saved student_search_results to {output_file}")
        return output_file

    
    def answer(self, query, k=5):
        metadata = self.index()
        minimal_search_results = build_retrieved_data(query, k, meta_data=metadata, id="1")
        print(generate_response(minimal_search_results))
    
    def answer_dataset(self, dataset_path, k=5):
        data_file = self.search_dataset(dataset_path, k)
        student_result_and_answers = call_llm_foreach_query(data_file)
        output_file = json_dump_search_and_answers(student_result_and_answers,
                                 "data/output/search_results_and_answer/" + dataset_path.split("/")[-2] + "/" + dataset_path.split("/")[-1]
                                )
        return output_file


def main():
    try:
        fire.Fire(Rag)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
