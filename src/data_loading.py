import pathlib
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

PERMITTED_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml"}
SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".venv"}
CODE_EXTENSIONS = {".py"}
data = pathlib.Path("data/raw/vllm-0.10.1")


def retrieve_files(path: pathlib.Path) -> list[pathlib.Path]:
    try:
        list_of_files: list[pathlib.Path] = []
        for item in path.rglob("*"):
            if (
                item.is_file()
                and item.suffix in PERMITTED_EXTENSIONS
                and not any(skip_dir in item.parts for skip_dir in SKIP_DIRS)
            ):
                list_of_files.append(item)
        return list_of_files
    except BaseException as e:
        print(e)
        exit(1)


def _build_splitters(chunk_size: int) -> tuple[RecursiveCharacterTextSplitter,
                                               RecursiveCharacterTextSplitter]:
    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=0,
        add_start_index=True,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        add_start_index=True,
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    return code_splitter, text_splitter


def load_and_split(list_of_files: list[pathlib.Path],
                   chunk_size: int
                   ) -> List[Document]:
    try:
        if chunk_size <= 0 or chunk_size > 2000:
            raise ValueError("chunk_size must be between 1 and 2000")
        documents: List[Document] = []
        code_splitter, text_splitter = _build_splitters(chunk_size)
        for file_path in list_of_files:
            try:
                loader = TextLoader(file_path, autodetect_encoding=True)
                loaded_documents = loader.load()
                splitter = (
                    code_splitter
                    if file_path.suffix in CODE_EXTENSIONS
                    else text_splitter
                )
                documents.extend(splitter.split_documents(loaded_documents))
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        return documents
    except BaseException as e:
        print(e)
        exit(20)
