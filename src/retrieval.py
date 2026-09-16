from functools import lru_cache
import bm25s
import chromadb


@lru_cache()
def load_bm25_index() -> bm25s.BM25:
    try:
        return bm25s.BM25.load("data/processed/bm25_index",
                               load_corpus=True,
                               mmap=True)
    except Exception as e:
        print(f"Error loading BM25 index: {e}")
        exit(4)


def retrieval(query: str, k: int) -> tuple[list[str], list[float]]:
    try:
        retriever = load_bm25_index()
        query_tokens = bm25s.tokenize(query)
        results, scores = retriever.retrieve(query_tokens,
                                             corpus=retriever.corpus,
                                             k=k)
        return results, scores
    except Exception as e:
        print(f"Error during retrieval: {e}")
        exit(4)


@lru_cache()
def load_chroma_collection() -> chromadb.api.models.Collection:
    try:
        client = chromadb.PersistentClient(path="data/processed/chroma_index")
        return client.get_collection(name="rag_collection")
    except Exception as e:
        print(f"Error loading Chroma collection: {e}")
        exit(4)


def chromadb_retrieval(query: str, k: int) -> list[dict[str, str]]:
    try:
        collection: chromadb.api.models.Collection = load_chroma_collection()
        return_results: list[dict[str, str]] = collection.query(
            query_texts=[query],
            n_results=k
            )
        return return_results
    except Exception as e:
        print(f"Error during retrieval: {e}")
        exit(4)
