from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """Identify a source file and the character range of a retrieved chunk."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question that does not yet have an answer."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its source references and answer text."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a dataset containing answered and unanswered questions."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Store a question and the sources retrieved for it."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Store retrieved sources together with a generated answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """Store search results for multiple questions and the requested count."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Store answers alongside results for multiple questions."""
    search_results: List[MinimalAnswer]
    k: int
