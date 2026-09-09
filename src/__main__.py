import argparse
import time
<<<<<<< HEAD
from search_dataset import retrieve_questions, review_chromadb_results, review_results, retrieve_question_id
=======
from search_dataset import retrieve_questions, retrieve_question_id
>>>>>>> b96b6db (need to fix pipline)
from indexing import index_files, chromadb_indexing
from llm_call import call_llm_foreach_query, json_dump_search_and_answers, json_dump_search_results
from build_retrieved_data import total_search_results, total_chromadb_search_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_chunk_size", type=int, default=2000)
    parser.add_argument("--dataset_path", type=str, default="data/datasets/AnsweredQuestions/dataset_docs_public.json")
    args = parser.parse_args()
    output_file = "data/output/search_results/".join(args.dataset_path.split("/")[2])
    start_time = time.time()
    #indexing the data files
    meta_data = index_files(args.max_chunk_size)
    # meta_data = chromadb_indexing(args.max_chunk_size)
    end_time = time.time()
    print(f"Indexing complete in {end_time - start_time:.2f} seconds. You can now use the BM25 retriever for searching.")
    #retrieving the questions from the json file
    questions = retrieve_questions(args.dataset_path)
    question_ids = retrieve_question_id(args.dataset_path)
    # retrieving the relevent data for each query (RA ANA LI KANTB HACHI MACHI AI AW9S) 
<<<<<<< HEAD
    # search_results = total_search_results(questions, question_ids, args.k, meta_data=meta_data) # bm25 search results
=======
    search_results = total_search_results(questions, question_ids, args.k, meta_data=meta_data) # bm25 search results
>>>>>>> b96b6db (need to fix pipline)
    # search_results = total_chromadb_search_results(questions, question_ids, args.k, meta_data=meta_data) # chromadb search results
    #sending the search results to the llm
    # start = time.time()
    # student_result_and_answers = call_llm_foreach_query(search_results)
    # end = time.time()
    # print(f"\nTime taken: {end - start:.2f}")
    # json_dump_search_and_answers(student_result_and_answers, "data/output/search_results_and_answer/output.json")
<<<<<<< HEAD
    # json_dump_search_results(student_result_and_answers, "data/output/search_results/output.json")
    review_results("datasets_public/public/AnsweredQuestions/dataset_docs_public.json",question_ids, args.k, meta_data)
    # review_chromadb_results("datasets_public/public/AnsweredQuestions/dataset_docs_public.json",question_ids, args.k, meta_data)
=======
    json_dump_search_results(search_results, "data/output/search_results/output.json")
    # review_results("data/datasets/AnsweredQuestions/dataset_docs_public.json",question_ids, args.k, meta_data)
    # review_chromadb_results("data/datasets/AnsweredQuestions/dataset_docs_public.json",question_ids, args.k, meta_data)
>>>>>>> b96b6db (need to fix pipline)

if __name__ == "__main__":
    main()
