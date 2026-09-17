from .data_models import (MinimalSearchResults,
                          MinimalSource,
                          StudentSearchResults)
from typing import List, Any
from .retrieval import chromadb_retrieval, retrieval
from functools import lru_cache
from collections import defaultdict


@lru_cache(maxsize=128)
def cached_retrieval(query: str, max_k: int = 50) -> Any:
    """Cache BM25 retrieval results for repeated queries.

    Args:
        query: Search query to execute.
        max_k: Maximum number of results to retrieve.

    Returns:
        The BM25 retrieval results and scores.
    """
    return retrieval(query, max_k)


@lru_cache(maxsize=128)
def cached_chromadb_retrieval(query: str, max_k: int = 50) -> Any:
    """Cache ChromaDB retrieval results for repeated queries.

    Args:
        query: Search query to execute.
        max_k: Maximum number of results to retrieve.

    Returns:
        The ChromaDB retrieval response.
    """
    return chromadb_retrieval(query, max_k)


def build_chromadb_retrieved_data(query: str, k: int, id: str = "") -> Any:
    """Build structured search results from ChromaDB metadata.

    Args:
        query: Search query used to find sources.
        k: Number of sources to include.
        id: Identifier for the question associated with the query.

    Returns:
        Search results containing the retrieved source locations.
    """
    minimal_search_results = MinimalSearchResults(
        question_id=id, question=query, retrieved_sources=[]
    )
    results = cached_chromadb_retrieval(query, k)
    for data in results["metadatas"][0]:
        minimal_search_results.retrieved_sources.append(
            MinimalSource(
                file_path=data["file_path"],
                first_character_index=data["start"],
                last_character_index=data["end"],
            )
        )
    return minimal_search_results


def build_retrieved_data(query: str, k: int, id: str = "") -> Any:
    """Build structured search results from BM25 records.

    Args:
        query: Search query used to find sources.
        k: Number of sources to include.
        id: Identifier for the question associated with the query.

    Returns:
        Search results containing the retrieved source locations.
    """
    minimal_search_results = MinimalSearchResults(
        question_id=id, question=query, retrieved_sources=[]
    )
    results, scores = cached_retrieval(query, max_k=50)

    for record in results[0][:k]:
        minimal_search_results.retrieved_sources.append(
            MinimalSource(
                file_path=record["file_path"],
                first_character_index=record["start"],
                last_character_index=record["end"],
            )
        )
    return minimal_search_results


def total_search_results(
    queries: List[str], question_ids: List[str], k: int
) -> StudentSearchResults:
    """Retrieve BM25 results for a collection of questions.

    Args:
        queries: Questions to search for.
        question_ids: Identifiers corresponding to the questions.
        k: Number of sources to retrieve per question.

    Returns:
        BM25 results grouped by question.
    """
    student_search_results = StudentSearchResults(search_results=[], k=k)
    for query, id in zip(queries, question_ids):
        search_result = build_retrieved_data(query, k, id=id)
        student_search_results.search_results.append(search_result)
    return student_search_results


def total_chromadb_search_results(
    queries: List[str], question_ids: List[str], k: int
) -> StudentSearchResults:
    """Retrieve ChromaDB results for a collection of questions.

    Args:
        queries: Questions to search for.
        question_ids: Identifiers corresponding to the questions.
        k: Number of sources to retrieve per question.

    Returns:
        ChromaDB results grouped by question.
    """
    student_search_results = StudentSearchResults(search_results=[], k=k)
    for query, id in zip(queries, question_ids):
        search_result = build_chromadb_retrieved_data(query, k, id=id)
        student_search_results.search_results.append(search_result)
    return student_search_results


def combine_search_results(
    bm25_results: StudentSearchResults,
    chromadb_results: StudentSearchResults,
) -> StudentSearchResults:
    """Combine BM25 and ChromaDB results while removing duplicate sources.

    Args:
        bm25_results: Results produced by the BM25 retriever.
        chromadb_results: Results produced by the ChromaDB retriever.

    Returns:
        Combined results in retriever order with unique source locations.
    """

    combined = StudentSearchResults(search_results=[], k=bm25_results.k)
    for bm25_result, chromadb_result in zip(
        bm25_results.search_results, chromadb_results.search_results
    ):
        combined_result = MinimalSearchResults(
            question_id=bm25_result.question_id,
            question=bm25_result.question,
            retrieved_sources=[],
        )
        seen = set()
        for source in bm25_result.retrieved_sources + \
                chromadb_result.retrieved_sources:
            source_id = (
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )
            if source_id not in seen:
                seen.add(source_id)
                combined_result.retrieved_sources.append(source)
        combined.search_results.append(combined_result)
    return combined


def reciprocal_rank_fusion(
    bm25_results: StudentSearchResults,
    chromadb_results: StudentSearchResults,
    k: int = 60,
) -> List[List[MinimalSource]]:
    """Fuse per-query BM25 and ChromaDB rankings using RRF.

    Args:
        bm25_results: Results produced by the BM25 retriever.
        chromadb_results: Results produced by the ChromaDB retriever.
        k: RRF ranking constant used to calculate source scores.

    Returns:
        One ranked list of sources per query, with the best source first.
    """
    fused_per_query = []

    for bm25_result, chroma_result in zip(
        bm25_results.search_results, chromadb_results.search_results
    ):
        scores: dict[tuple, float] = defaultdict(float)
        source_by_id: dict[tuple, MinimalSource] = {}

        for rank, source in enumerate(bm25_result.retrieved_sources, start=1):
            sid = (
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )
            scores[sid] += 1.0 / (k + rank)
            source_by_id[sid] = source

        for rank, source in enumerate(chroma_result.retrieved_sources,
                                      start=1):
            sid = (
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )
            scores[sid] += 1.0 / (k + rank)
            source_by_id[sid] = source

        ranked_ids = sorted(scores, key=lambda sid: scores[sid], reverse=True)
        fused_per_query.append([source_by_id[sid] for sid in ranked_ids])
    return fused_per_query
