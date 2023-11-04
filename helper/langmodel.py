from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
import json
from dotenv import load_dotenv

load_dotenv()




model =ChatOpenAI(
    model_name="gpt-4",
)

class InterviewQuestions(BaseModel):
    questions: List[str] = Field(description="List of interview questions based on topic and programming language")

class AnswerEvaluation(BaseModel):
    score: float = Field(description="Score based on the number of questions answered correctly")
    correct_answers: List[dict] = Field(description="Model generated answers for the questions")



question_parser = PydanticOutputParser(pydantic_object=InterviewQuestions)
answer_parser = PydanticOutputParser(pydantic_object=AnswerEvaluation)


prompt = PromptTemplate(
    template="Generate 10 {level} level interview questions based on the topic {topic} and the programming language {language}.\n{format_instructions}\n",
    input_variables=["topic", "language", "level"],
    partial_variables={"format_instructions": question_parser.get_format_instructions()},
)


answer_prompt = PromptTemplate(
    template="Evaluate the following answers to the given questions. For only and only the  question that you think has been answered incorrectly, provide the correct answer.\n{format_instructions}\nQuestions: {questions}\nAnswers: {answers}\n",
    input_variables=["questions", "answers"],
    partial_variables={"format_instructions": answer_parser.get_format_instructions()},
)

def evaluate_answers(qa_pairs: List[dict]) -> dict:
    questions = [qa_pair["question"] for qa_pair in qa_pairs]
    answers = [qa_pair["answer"] for qa_pair in qa_pairs]

    prompt_and_model = answer_prompt | model
    output = prompt_and_model.invoke({"questions": questions, "answers": answers})
    evaluation = answer_parser.invoke(output)

    return {"score": evaluation.score, "correct_answers": evaluation.correct_answers}

def generate_questions(topic: str, languages: List[str], level: str) -> dict:
    all_questions = []

    for language in languages:
        prompt_and_model = prompt | model
        output = prompt_and_model.invoke({"topic": topic, "language": language, "level": level})
        questions = question_parser.invoke(output)
        all_questions.extend(questions.questions)

    return {"questions": all_questions}

# Usage
topic = "data science"
languages = ["Python", "SQL", "Maths"]
level = "beginner"

questions_dict = generate_questions(topic, languages, level)
print(json.dumps(questions_dict, indent=4))





qa_pairs = [
    {
        "question": "What is Data Science and how does it differ from traditional statistics?",
        "answer": "Data Science is a multidisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It differs from traditional statistics in that it incorporates techniques and theories from broader fields like mathematics, statistics, information science, and computer science. Traditional statistics, on the other hand, is more focused on data analysis and interpretation in isolation."
    },
    {
        "question": "Can you briefly explain the data science lifecycle?",
        "answer": "The data science lifecycle typically includes the following steps: 1) Business Understanding - defining the problem and desired outcomes. 2) Data Mining - gathering the required data. 3) Data Cleaning - removing inconsistencies and inaccuracies in the data. 4) Data Exploration - using statistical techniques to identify patterns or anomalies in the data. 5) Feature Engineering - modifying or creating new variables to better represent the underlying data patterns. 6) Predictive Modeling - creating a statistical model for prediction. 7) Data Visualization - visualizing the data and model outcomes. 8) Deployment - implementing the model into production. 9) Monitoring - tracking the model's performance over time."
    },
    {
        "question": "What are some common libraries in Python used for data science and what are their uses?",
        "answer": "Some common Python libraries used in data science include: 1) NumPy - used for numerical computations and handling arrays. 2) Pandas - used for data manipulation and analysis. 3) Matplotlib - used for data visualization. 4) Scikit-learn - used for machine learning and statistical modeling. 5) TensorFlow and PyTorch - used for deep learning."
    },
    {
        "question": "What is a pandas DataFrame in Python and how is it used in data science?",
        "answer": "A pandas DataFrame is a two-dimensional, size-mutable, heterogeneous tabular data structure with labeled axes (rows and columns). It can be thought of as a dictionary-like container for Series objects. In data science, it is used for data manipulation and analysis. It provides various functionalities like slicing, indexing, joining, merging, reshaping, etc."
    },
    {
        "question": "Can you describe what 'data cleaning' is and why it's important in data science?",
        "answer": "Data cleaning, also known as data cleansing or data scrubbing,we use it to add value to the data by adding stopwords to make it longer for our model to understand"
    }
]


evaluation_dict = evaluate_answers(qa_pairs)
print(json.dumps(evaluation_dict, indent=4))