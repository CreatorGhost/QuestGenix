from typing import List
from pydantic import BaseModel


class QuestionAnswerPair(BaseModel):
    question: str
    answer: str

class AnswerEvaluationRequest(BaseModel):
    qa_pairs: List[QuestionAnswerPair]

class AnswerEvaluationResponse(BaseModel):
    score: float
    correct_answers: List[QuestionAnswerPair]