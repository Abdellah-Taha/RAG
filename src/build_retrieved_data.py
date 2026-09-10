from data_models import MinimalSearchResults, MinimalSource, StudentSearchResults
from typing import List
from retrieval import chromadb_retrieval, retrieval
from functools import lru_cache
import time

@lru_cache(maxsize=128)
def cached_retrieval(query: str, max_k=50):
    return retrieval(query, max_k)

@lru_cache(maxsize=128)
def cached_chromadb_retrieval(query: str, max_k=50):
    return chromadb_retrieval(query, max_k)

def build_chromadb_retrieved_data(query: str, k: int, meta_data: List[dict], id=""):
    minimal_search_results = MinimalSearchResults(question_id=id, question=query, retrieved_sources=[])
    results = cached_chromadb_retrieval(query, max_k=50)
    meta_by_chunk_id = {m["chunk_id"]: m for m in meta_data} # create a mapping from chunk_id to meta_data for quick lookup

    hit_ids = results["ids"][0] # get the list of hit ids for the first query
    limit = min(k, len(hit_ids))

    for i in range(limit):
        m = meta_by_chunk_id[hit_ids[i]]
        minimal_search_results.retrieved_sources.append(MinimalSource(
            file_path=m["file_path"],
            first_character_index=m["start"],
            last_character_index=m["end"],
        ))
    return minimal_search_results



def build_retrieved_data(query: str, k: int, meta_data: List[dict], id=""):
    minimal_search_results = MinimalSearchResults(question_id=id, question=query, retrieved_sources=[])
    results, scores = cached_retrieval(query, max_k=50)
    
    limit = min(k, len(results[0]))
    
    for i in range(limit):
        doc_idx = results[0][i]
        minimal_source = MinimalSource(
            file_path=meta_data[doc_idx]['file_path'],
            first_character_index=meta_data[doc_idx]['start'],
            last_character_index=meta_data[doc_idx]['end'],
        )
        minimal_search_results.retrieved_sources.append(minimal_source)
        
    return minimal_search_results

def total_search_results(queries: List[str],question_ids: List[str], k: int, meta_data: List[dict]) -> StudentSearchResults:
    student_search_results = StudentSearchResults(search_results=[], k=k)
    for query, id in zip(queries, question_ids):
        search_result = build_retrieved_data(query, id, k, meta_data)
        student_search_results.search_results.append(search_result)
    return student_search_results

def total_chromadb_search_results(queries: List[str], question_ids: List[str], k: int, meta_data: List[dict]) -> StudentSearchResults:
    student_search_results = StudentSearchResults(search_results=[], k=k)
    for query, id in zip(queries, question_ids):
        search_result = build_chromadb_retrieved_data(query, id, k, meta_data)
        student_search_results.search_results.append(search_result)
    return student_search_results
