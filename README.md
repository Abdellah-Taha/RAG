# RAG against the machine

A Retrieval-Augmented Generation system built over the [vLLM](https://github.com/vllm-project/vllm) codebase
(`vllm-0.10.1`). Given a natural-language question about vLLM's code or documentation, it retrieves the most
relevant source chunks and generates a grounded answer with a small local LLM.

42 curriculum project — built without high-level RAG frameworks (no LangChain retrievers, no LlamaIndex); indexing,
chunking, and retrieval are implemented directly.

## Overview

- **Corpus**: the vLLM `0.10.1` source tree — Python code and Markdown/text documentation.
- **Chunking**: two dedicated strategies, both built on `langchain_text_splitters.RecursiveCharacterTextSplitter`
  - `.py` files → a Python-aware splitter (`Language.PYTHON`) that respects function/class boundaries.
  - Docs/text files (`.md`, `.rst`, `.txt`, `.json`, `.yaml`, `.yml`) → a generic recursive character splitter.
  - Chunk size is configurable and capped at 2000 characters.
- **Retrieval**: hybrid search combining
  - **BM25** (via [`bm25s`](https://github.com/xhluca/bm25s)) — sparse, lexical retrieval.
  - **Dense embeddings** (via [ChromaDB](https://www.trychroma.com/)) — semantic retrieval.
  - Results from both are combined with **Reciprocal Rank Fusion**.
- **Answer generation**: retrieved chunks are passed as context to `Qwen/Qwen3-0.6B` (via `transformers`) to produce
  a grounded answer.
- **Data models**: all inputs/outputs (questions, sources, search results, answers) are `pydantic` models
  (`src/data_models.py`), matching the schema expected by the grading moulinette.

## Requirements

- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Several GB of free disk space (PyTorch, transformers, and the `Qwen3-0.6B` weights are large — run this from a
  location with enough room)
- The vLLM `0.10.1` source archive available locally under `data/raw/vllm-0.10.1/` (not included in this repository
  — see [Data](#data) below)

## Installation

```bash
make install
```

This runs `uv sync`, creating a virtual environment and installing all dependencies declared in `pyproject.toml`
(pinned in `uv.lock`).

## Data

This repository intentionally does **not** track the vLLM source corpus, the generated indices, or generated
output/answer files — they are either large, or reproducible by running the commands below. Before indexing, place
the extracted vLLM `0.10.1` source tree at:

```
data/raw/vllm-0.10.1/
```

Running `index` (see below) will then populate `data/processed/` with the BM25 and Chroma indices.

## Usage

All commands are exposed through a `Fire`-based CLI, invoked as a package (`python -m src ...`) so that the
project's internal relative imports resolve correctly.

### Index the corpus

```bash
uv run python -m src index [--max_chunk_size 2000]
```

Walks `data/raw/vllm-0.10.1/`, splits every eligible file with the appropriate chunker, and builds both the BM25
index and the Chroma vector index under `data/processed/`.

### Search a single question

```bash
uv run python -m src search "<your question>" [--k 5]
```

Runs hybrid (BM25 + Chroma) retrieval for one query and prints the top `k` matching source locations. Indexes are
built automatically on first use if `data/processed/` doesn't exist yet.

### Search a full dataset

```bash
uv run python -m src search_dataset <dataset_path> [--k 10] [--save_directory data/output/search_results]
```

Runs retrieval for every question in a dataset file (e.g. `data/datasets/AnsweredQuestions/dataset_code_public.json`)
and writes a `StudentSearchResults` JSON file — the format the moulinette expects for recall evaluation.

### Answer a single question

```bash
uv run python -m src answer "<your question>" [--k 5]
```

Retrieves context for the question and prints a generated answer from `Qwen3-0.6B`.

### Answer a full dataset

```bash
uv run python -m src answer_dataset <student_search_results_path> [--save_directory data/output/search_results_and_answers]
```

Takes a previously generated `search_dataset` output and generates an answer for every question, writing a
`StudentSearchResultsAndAnswer` JSON file.

### Evaluate locally

```bash
uv run python -m src evaluate <student_search_results_path> <dataset_path>
```

Computes recall against a ground-truth dataset. Note: the *official* grading recall metric is computed by the
external moulinette binary, which this project never imports or calls directly — this command is for local sanity
checks only.

## Makefile targets

| Target  | Description |
|---------|--------------|
| `install` | `uv sync` — install/sync dependencies |
| `run`     | Run the CLI's default pipeline |
| `debug`   | Launch the CLI under `pdb` (`python -m pdb -m src`, so relative imports still resolve) |
| `clean`   | Remove `__pycache__` / `.mypy_cache` |
| `lint`    | `flake8` + `mypy` (non-strict: untyped defs still flagged, unknown imports ignored) |
| `lint-strict` | `flake8` + `mypy --strict` |

## Project layout

```
src/
├── __main__.py               # Fire CLI entry point (Rag class)
├── data_loading.py           # file discovery + chunking (code vs. docs splitters)
├── indexing.py                # BM25 + Chroma index construction
├── retrieval.py                # BM25 + Chroma query-time retrieval
├── build_retrieved_data.py     # retrieval orchestration + reciprocal rank fusion
├── search_dataset.py           # dataset-level search + local evaluation helpers
├── llm_call.py                  # Qwen3-0.6B loading + answer generation
└── data_models.py                # pydantic schemas (questions, sources, results, answers)
```

## Constraints this project targets

- Indexing the whole corpus in ≤ 5 minutes
- Retrieval over 200 questions in ≤ 90 seconds
- Recall@5 ≥ 80% on docs questions, ≥ 50% on code questions
- Chunks (and generation context) capped at 2000 characters

## Bonus work

- [x] Semantic embeddings
- [x] ChromaDB hybrid with BM25 via reciprocal rank fusion
- [x] Caching
- [ ] Incremental indexing (skips BM25 re-indexing when source files are unchanged, via mtime comparison)
- [ ] Local HTTP API

## Known limitations

- `answer_dataset` currently loads the full `Qwen3-0.6B` model per invocation; there is no batching across
  questions yet.

## Helpful link

- [RAG Against the Machine: Theory Guide](https://app.notion.com/p/RAG-Against-the-Machine-Theory-Guide-3dedc0786e998079acf2e8502fb65305)