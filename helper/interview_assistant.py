# File: interview_assistant.py

from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
import json
from dotenv import load_dotenv

load_dotenv()

class InterviewQuestions(BaseModel):
    questions: List[str] = Field(description="List of interview questions based on topic and programming language")

class AnswerEvaluation(BaseModel):
    score: float = Field(description="Score based on the number of questions answered correctly")
    correct_answers: List[dict] = Field(description="Model generated answers for the questions")

class InterviewAssistant:
    def __init__(self, model_name: str):
        self.model = ChatOpenAI(model_name=model_name)
        self.question_parser = PydanticOutputParser(pydantic_object=InterviewQuestions)
        self.answer_parser = PydanticOutputParser(pydantic_object=AnswerEvaluation)
        self.question_prompt = PromptTemplate(
            template="Generate 10 {level} level interview questions based on the topic {topic} and the programming language {language}.\n{format_instructions}\n",
            input_variables=["topic", "language", "level"],
            partial_variables={"format_instructions": self.question_parser.get_format_instructions()},
        )
        self.answer_prompt = PromptTemplate(
            template="Evaluate the following answers to the given questions. For only and only the  question that you think has been answered incorrectly, provide the correct answer.\n{format_instructions}\nQuestions: {questions}\nAnswers: {answers}\n",
            input_variables=["questions", "answers"],
            partial_variables={"format_instructions": self.answer_parser.get_format_instructions()},
        )

    def evaluate_answers(self, qa_pairs: List[dict]) -> dict:
        
        questions = [qa_pair.question for qa_pair in qa_pairs]
        answers = [qa_pair.answer for qa_pair in qa_pairs]
        prompt_and_model = self.answer_prompt | self.model
        output = prompt_and_model.invoke({"questions": questions, "answers": answers})
        output_dict = json.loads(output.content)
        
        # Transform correct_answers into the expected format
        correct_answers_dicts = [
            {"question": question, "answer": answer}
            for correct_answer_dict in output_dict['correct_answers']
            for question, answer in correct_answer_dict.items()
        ]
        
        # Return the transformed output
        return {
            "score": output_dict['score'],
            "correct_answers": correct_answers_dicts
        }

    def generate_questions(self, topic: str, languages: List[str], level: str) -> dict:
        all_questions = []
        for language in languages:
            prompt_and_model = self.question_prompt | self.model
            output = prompt_and_model.invoke({"topic": topic, "language": language, "level": level})
            questions = self.question_parser.invoke(output)
            all_questions.extend(questions.questions)
        return {"questions": all_questions}



# Usage
# assistant = InterviewAssistant(model_name="gpt-4")
# topic = "data science"
# languages = ["Python", "SQL", "Maths"]
# level = "beginner"
# questions_dict = assistant.generate_questions(topic, languages, level)
# print(json.dumps(questions_dict, indent=4))

# qa_pairs = [
#     # ... your question-answer pairs here ...
# ]
# evaluation_dict = assistant.evaluate_answers(qa_pairs)
# print(json.dumps(evaluation_dict, indent=4))