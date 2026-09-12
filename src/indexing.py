from functools import lru_cache
from typing import List
from data_loading import retrieve_files, load_and_split
from langchain_core.documents import Document
import bm25s, pathlib
import chromadb
import tqdm


BATCH_SIZE = 600
data_path = pathlib.Path("data/raw/vllm-0.10.1")

def check_update_on_files(data_path: pathlib.Path, processed_data_path: pathlib.Path):
    try:
        updated_files: List[pathlib.Path] = []
        raw_files: List[pathlib.Path] = list(data_path.glob("**/*"))
        processed_last_update = processed_data_path.stat().st_mtime if processed_data_path.exists() else 0
        for raw_file in raw_files:
            if raw_file.is_file():
                if raw_file.stat().st_mtime > processed_last_update:
                    updated_files.append(raw_file)
        if updated_files:
            print(f"Found {len(updated_files)} updated files. Proceeding with indexing.")
        return updated_files
    except Exception as e:
        print(f"Error while retrieving raw files: {e}")
        return False

def index_files(chunk_size: int) -> List[dict]:
    #check if the processed data already exists if it does and up to date then skip indexing
    #if the processed data does not exist index the entire corpus
    #if the data has been updated then re-load, re-chunk and re-tokenize only the updated files then re-index the entire data 
    
    try:
        processed_data_path = pathlib.Path("data/processed/bm25_index")
        updated_files = check_update_on_files(data_path, processed_data_path)
        sample = retrieve_files(data_path)
        documents: List[Document] = load_and_split(sample, chunk_size)
        content: List[str] = []
        metadata: List[dict] = []
        for document in tqdm.tqdm(documents, desc="BM25 indexing"):
            content.append(document.page_content)
            metadata.append({
            "file_path": document.metadata["source"],
            "start": document.metadata["start_index"],
            "end": document.metadata["start_index"] + len(document.page_content),
            })
        if not processed_data_path.exists():
            #normal indexing of the entire corpus
                    
            corpus = bm25s.tokenize(content)
            indexer = bm25s.BM25()
            indexer.index(corpus)
            indexer.save("data/processed/bm25_index")
        elif updated_files:
            # re chunking only the updated files in the list
            ...
        else:
            #skiping indexing as the processed data is up to date
            ...
        return metadata
    except Exception as e:
        print(f"Error during indexing: {e}")
        exit(3)



@lru_cache(maxsize=128)
def chromadb_indexing(chunk_size: int):
    try:
        sample = retrieve_files(data_path)  
        documents: List[Document] = load_and_split(sample, chunk_size)
        content: List[str] = []
        metadata: List[dict] = []
        chunk_ids: List[str] = []

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
