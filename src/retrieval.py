from functools import lru_cache
import bm25s
import chromadb

@lru_cache()
def load_bm25_index():
    try:
        retriever = bm25s.BM25.load("data/processed/bm25_index")
        return retriever
    except Exception as e:
        print(f"Error loading BM25 index: {e}")
        exit(4)
        
def retrieval(query: str, k: int):
    try:
        retriever = load_bm25_index()
        query_tokens = bm25s.tokenize(query)
        results, scores = retriever.retrieve(query_tokens, k=k)
        return results, scores
    except Exception as e:
        print(f"Error during retrieval: {e}, line: {e.__traceback__.tb_lineno}")
        exit(4)
        

def chromadb_retrieval(query: str, k: int):
    try:
        indexed_data_path = "data/processed/chroma_index"
        client = chromadb.PersistentClient(path=indexed_data_path)
        collection = client.get_collection(name="rag_collection")
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        return results
    except Exception as e:
        print(f"Error during retrieval: {e}, line: {e.__traceback__.tb_lineno}")
        exit(4)
