from typing import List
from pydantic import BaseModel


class AnswerEvaluationRequest(BaseModel):
    qa_pairs: List[dict]

class AnswerEvaluationResponse(BaseModel):
    score: float
    correct_answers: List[dict]