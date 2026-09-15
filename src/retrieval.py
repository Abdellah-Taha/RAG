from functools import lru_cache
import bm25s
import chromadb

@lru_cache()
def load_bm25_index():
    try:
        return bm25s.BM25.load("data/processed/bm25_index", load_corpus=True, mmap=True)
    except Exception as e:
        print(f"Error loading BM25 index: {e}")
        exit(4)

def retrieval(query: str, k: int):
    try:
        retriever = load_bm25_index()
        query_tokens = bm25s.tokenize(query)
        results, scores = retriever.retrieve(
            query_tokens, corpus=retriever.corpus, k=k
        )
        return results, scores
    except Exception as e:
        print(f"Error during retrieval: {e}, line: {e.__traceback__.tb_lineno}")
        exit(4)
        

@lru_cache()
def load_chroma_collection():
    try:
        client = chromadb.PersistentClient(path="data/processed/chroma_index")
        return client.get_collection(name="rag_collection")
    except Exception as e:
        print(f"Error loading Chroma collection: {e}")
        exit(4)

def chromadb_retrieval(query: str, k: int):
    try:
        collection = load_chroma_collection()
        return collection.query(query_texts=[query], n_results=k)
    except Exception as e:
        print(f"Error during retrieval: {e}, line: {e.__traceback__.tb_lineno}")
        exit(4)
