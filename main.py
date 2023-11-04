from fastapi import FastAPI, Body
from models.question import QuestionGenerationRequest, QuestionGenerationResponse
from models.answer import AnswerEvaluationRequest, AnswerEvaluationResponse
from helper.interview_assistant import InterviewAssistant


app = FastAPI()

assistant = InterviewAssistant(model_name="gpt-4")


@app.post("/generate_questions/", response_model=QuestionGenerationResponse)
async def generate_questions(request: QuestionGenerationRequest):
    questions_dict = assistant.generate_questions(request.topic, request.languages, request.level)
    return questions_dict

@app.post("/evaluate_answers/", response_model=AnswerEvaluationResponse)
async def evaluate_answers(request: AnswerEvaluationRequest):
    evaluation_dict = assistant.evaluate_answers(request.qa_pairs)
    return evaluation_dict

