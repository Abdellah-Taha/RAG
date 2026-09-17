from functools import lru_cache
from typing import List
from .data_models import (
    StudentSearchResults,
    MinimalSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
from pathlib import Path
from typing import Any


def strip_thinking(text: str) -> str:
    """Remove hidden reasoning tags from generated model output.

    Args:
        text: Raw text returned by the language model.

    Returns:
        Text with ``<think>`` blocks removed and whitespace trimmed.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class Llm:
    """Generate answers with a configured Hugging Face model."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """Load the tokenizer and model identified by ``model_name``.

        Args:
            model_name: Hugging Face model identifier to load.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 150) -> Any:
        """Generate a concise answer for a prompt.

        Args:
            prompt: User prompt containing the question and context.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The generated answer with hidden reasoning removed.
        """
        message = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. \
Answer the question using only the "
                    "contextual information provided. \
Give a direct, concise answer — "
                    "state the fact directly without \
restating the question or explaining "
                    "your reasoning. If the context \
includes a specific endpoint, command, "
                    "or value, quote it exactly."
                    "if the answer is not contained within"
                    "the context, say \"I don't know\"."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            message, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        input_text = self.tokenizer(
            prompt_text, return_tensors="pt",
            padding=True,
            truncation=True
        )

        input_length = input_text["input_ids"].shape[1]

        output = self.model.generate(
            **input_text,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = output[0][input_length:]
        output_text = self.tokenizer.decode(generated_tokens,
                                            skip_special_tokens=True)
        return strip_thinking(output_text)


@lru_cache()
def call_llm() -> Any:
    """Return the cached default language-model wrapper.

    Returns:
        A lazily initialized ``Llm`` instance.
    """
    return Llm()


def extract_text_from_context(context: MinimalSearchResults) -> Any:
    """Yield text ranges referenced by a search result.

    Args:
        context: Search result containing source paths and character ranges.

    Yields:
        The selected text from each retrieved source.
    """
    for path in context.retrieved_sources:
        with open(path.file_path, "r", encoding="utf-8") as f:
            text = f.read()
            yield text[path.first_character_index: path.last_character_index]


def generate_response(context: MinimalSearchResults,
                      max_new_tokens: int = 150) -> Any:
    """Generate an answer using the retrieved context for one question.

    Args:
        context: Question and source ranges used to construct the prompt.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The generated answer text.
    """
    llm = call_llm()
    result: List[str] = []
    for text in extract_text_from_context(context):
        result.append(text)
    super_prompt = (
        f"{context.question}\n\nContext:\n" + "\n".join(result) + "\nResponse:"
    )
    return str(llm.generate(super_prompt, max_new_tokens))


def call_llm_foreach_query(context: StudentSearchResults) -> Any:
    """Generate and print an answer for every question in a result set.

    Args:
        context: Search results containing the questions to answer.

    Returns:
        Generated answer strings in the same order as the input questions.
    """
    responses = []
    for i, result in enumerate(context.search_results):
        response_text = generate_response(result)
        responses.append(response_text)

        print(f"Question: {result.question}")
        print(f"response: {responses[i]}")
        print("=======================================================")
    return responses


def create_student_search_results_and_answer(
    context: StudentSearchResults, responses: List[str]
) -> StudentSearchResultsAndAnswer:
    """Attach generated responses to their corresponding search results.

    Args:
        context: Search results for the answered questions.
        responses: Answers in the same order as ``context.search_results``.

    Returns:
        Search results enriched with generated answers.
    """
    minimal_answers: List[MinimalAnswer] = []
    for i, result in enumerate(context.search_results):
        minimal_answer = MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=responses[i],
        )
        minimal_answers.append(minimal_answer)

    return StudentSearchResultsAndAnswer(search_results=minimal_answers,
                                         k=context.k)


def json_dump_search_results(answers: StudentSearchResults,
                             output_file: str
                             ) -> Any:
    """Serialize search results as formatted JSON.

    Args:
        answers: Search results to serialize.
        output_file: Destination path for the JSON document.

    Returns:
        The output file path.
    """
    search_results_list: dict[str, Any] = {"search_results": [],
                                           "k": answers.k}
    for data in answers.search_results:
        data_dict = {
            "question_id": data.question_id,
            "question": data.question,
            "retrieved_sources": [
                {
                    "file_path": source.file_path,
                    "first_character_index": source.first_character_index,
                    "last_character_index": source.last_character_index,
                }
                for source in data.retrieved_sources
            ],
        }
        search_results_list["search_results"].append(data_dict)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(search_results_list, f, ensure_ascii=False, indent=4)
    return output_file


def json_dump_search_and_answers(
    answers: StudentSearchResultsAndAnswer, output_file: str
) -> Any:
    """Serialize search results and generated answers as formatted JSON.

    Args:
        answers: Search results and answers to serialize.
        output_file: Destination path for the JSON document.

    Returns:
        The output file path.
    """
    search_results_list: dict[str, Any] = {"search_results": [],
                                           "k": answers.k}
    for data in answers.search_results:
        data_dict = {
            "question_id": data.question_id,
            "question": data.question,
            "retrieved_sources": [
                {
                    "file_path": source.file_path,
                    "first_character_index": source.first_character_index,
                    "last_character_index": source.last_character_index,
                }
                for source in data.retrieved_sources
            ],
            "answer": data.answer,
        }
        search_results_list["search_results"].append(data_dict)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(search_results_list, f, ensure_ascii=False, indent=4)
