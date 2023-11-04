from typing import List
from pydantic import BaseModel

class QuestionGenerationRequest(BaseModel):
    topic: str
    languages: List[str]
    level: str

class QuestionGenerationResponse(BaseModel):
    questions: List[str]