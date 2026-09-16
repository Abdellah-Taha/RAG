from typing import List, Any
from .data_loading import retrieve_files, load_and_split
from langchain_core.documents import Document
import bm25s
import pathlib
import chromadb
import tqdm

BATCH_SIZE = 600
data_path = pathlib.Path("data/raw/vllm-0.10.1")


def files_mtime(folder: pathlib.Path) -> Any:
    try:
        file_times = (
            f.stat().st_mtime for f in folder.rglob("*") if f.is_file()
            )
        return max(file_times, default=folder.stat().st_mtime)
    except BaseException as e:
        print(e)
        exit(67)


def get_metadata(documents: List[Document]) -> Any:
    try:
        content: List[str] = []
        metadata: List[dict] = []
        for document in tqdm.tqdm(documents, desc="BM25 indexing"):
            content.append(document.page_content)
            metadata.append(
                {
                    "file_path": document.metadata["source"],
                    "start": document.metadata["start_index"],
                    "end": document.metadata["start_index"]
                    + len(document.page_content),
                }
            )
        return metadata, content
    except BaseException as e:
        print(e)
        exit(1)


def index_files(chunk_size: int) -> Any:
    try:
        sample = retrieve_files(data_path)
        documents: List[Document] = load_and_split(sample, chunk_size)
        metadata, content = get_metadata(documents)
        records = [
            {**meta, "text": text} for meta, text in zip(metadata, content)
            ]
        corpus = bm25s.tokenize(content)
        indexer = bm25s.BM25()
        indexer.index(corpus)
        indexer.save("data/processed/bm25_index", corpus=records)
    except BaseException as e:
        print(f"Error during indexing: {e}")
        exit(3)


def chromadb_indexing(chunk_size: int) -> Any:
    try:
        sample = retrieve_files(data_path)
        documents: List[Document] = load_and_split(sample, chunk_size)
        content: List[str] = []
        metadata: List[dict] = []
        chunk_ids: List[str] = []

        for document in documents:
            content.append(document.page_content)

            current_chunk_id = (
                f"{document.metadata['source']}\
_{document.metadata['start_index']}"
            )
            chunk_ids.append(current_chunk_id)

            metadata.append(
                {
                    "file_path": document.metadata["source"],
                    "start": document.metadata["start_index"],
                    "end": document.metadata["start_index"]
                    + len(document.page_content),
                    "chunk_id": current_chunk_id,
                }
            )

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Duplicate chunk_id found in metadata. \
Each chunk must have a unique chunk_id."
            )

        client = chromadb.PersistentClient(path="data/processed/chroma_index")
        collection = client.get_or_create_collection(name="rag_collection")

        for i in tqdm.tqdm(
            range(0, len(content), BATCH_SIZE), desc="chromadb indexing"
        ):
            batch_content = content[i: i + BATCH_SIZE]
            batch_metadata = metadata[i: i + BATCH_SIZE]
            batch_ids = chunk_ids[i: i + BATCH_SIZE]

            collection.add(
                documents=batch_content,
                metadatas=batch_metadata,
                ids=batch_ids
            )
        return metadata
    except BaseException as e:
        print(f"Error during ChromaDB indexing: {e}")
        exit(3)
