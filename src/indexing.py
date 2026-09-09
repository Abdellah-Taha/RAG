from functools import lru_cache
from typing import List
from data_loading import retrieve_files, load_and_split
from langchain_core.documents import Document
import bm25s, pathlib
import chromadb
import tqdm


BATCH_SIZE = 600
data_path = pathlib.Path("data/raw/vllm-0.10.1")

def index_files(chunk_size: int) -> List[dict]:
    try:
        sample = retrieve_files(data_path)
        documents: List[Document] = load_and_split(sample, chunk_size)
        content = []
        metadata = []
        for document in tqdm.tqdm(documents, desc="BM25 indexing"):
            content.append(document.page_content)
            metadata.append({
            "file_path": document.metadata["source"],
            "start": document.metadata["start_index"],
            "end": document.metadata["start_index"] + len(document.page_content),
            })
            
        corpus = bm25s.tokenize(content)
        indexer = bm25s.BM25()
        indexer.index(corpus)
        indexer.save("data/processed/bm25_index")
        return metadata
    except Exception as e:
        print(f"Error during indexing: {e}")
        exit(3)

@lru_cache(maxsize=128)
def chromadb_indexing(chunk_size: int):
    try:
        sample = retrieve_files(data_path)  
        documents: List = load_and_split(sample, chunk_size)
        content = []
        metadata = []
        chunk_ids = []

        for document in documents:
            content.append(document.page_content)
            
            current_chunk_id = f"{document.metadata['source']}_{document.metadata['start_index']}"
            chunk_ids.append(current_chunk_id)
            
            metadata.append({
                "file_path": document.metadata["source"],
                "start": document.metadata["start_index"],
                "end": document.metadata["start_index"] + len(document.page_content),
                "chunk_id": current_chunk_id
            })

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate chunk_id found in metadata. Each chunk must have a unique chunk_id.")

        client = chromadb.PersistentClient(path="data/processed/chroma_index")
        collection = client.get_or_create_collection(name="rag_collection")

        for i in tqdm.tqdm(range(0, len(content), BATCH_SIZE), desc="chromadb indexing"):
            batch_content = content[i:i + BATCH_SIZE]
            batch_metadata = metadata[i:i + BATCH_SIZE]
            batch_ids = chunk_ids[i:i + BATCH_SIZE]
            
            collection.add(
                documents=batch_content,
                metadatas=batch_metadata,
                ids=batch_ids
            )
        return metadata
    except Exception as e:
        print(f"Error during ChromaDB indexing: {e}")
        exit(3)

import time
def main():
    chunk_size = 2000
    start_time = time.time()
    index_files(chunk_size)
    end_time = time.time()
    print(f"BM25 indexing complete in {end_time - start_time:.2f} seconds.")
    start_time = time.time()
    chromadb_indexing(chunk_size)
    end_time = time.time()
    print(f"chromadb indexing complete in {end_time - start_time:.2f} seconds.")
if __name__ == "__main__":
    main()
