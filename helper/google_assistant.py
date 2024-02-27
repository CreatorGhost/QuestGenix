# File: interview_assistant.py

from typing import List
import tempfile
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
import json
from dotenv import load_dotenv
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
import re
import os
import vertexai
from langchain_google_vertexai import VertexAI

load_dotenv()




# Construct the service account info JSON
service_account_info = {
    "type": os.getenv("SERVICE_ACCOUNT_TYPE"),
    "project_id": os.getenv("SERVICE_ACCOUNT_PROJECT_ID"),
    "private_key_id": os.getenv("SERVICE_ACCOUNT_PRIVATE_KEY_ID"),  
    "private_key": os.getenv("SERVICE_ACCOUNT_PRIVATE_KEY").replace('\\n', '\n'),  # The actual private key
    "client_email": os.getenv("SERVICE_ACCOUNT_CLIENT_EMAIL"),
    "client_id": os.getenv("SERVICE_ACCOUNT_CLIENT_ID"),
    "auth_uri": os.getenv("SERVICE_ACCOUNT_AUTH_URI"),
    "token_uri": os.getenv("SERVICE_ACCOUNT_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("SERVICE_ACCOUNT_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("SERVICE_ACCOUNT_CLIENT_X509_CERT_URL"),
    "universe_domain": os.getenv("SERVICE_ACCOUNT_UNIVERSE_DOMAIN")
}

# Create a temporary file to hold the service account JSON
with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
    # Write the service account JSON to the temporary file
    json.dump(service_account_info, temp_file)
    # Get the path of the temporary file
    temp_file_path = temp_file.name

# Set the path to the temporary file in the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path


PROJECT_ID = "active-cirrus-413708" 
REGION = "us-central1" 
vertexai.init(project=PROJECT_ID, location=REGION)




def extract_json(string):
    # Use regular expression to remove leading and trailing triple backticks
    string = re.sub(r'^```|```$', '', string)

    # Split the string at any occurrence of the word 'json' (case-insensitive)
    parts = re.split(r'json', string, flags=re.IGNORECASE)

    # Check if at least one part remains after splitting
    if len(parts) < 2:
        return None

    # Use the second part (assuming the first part contains unnecessary text)
    json_data = parts[1].strip()  # Strip any leading/trailing whitespace

    # Try to parse the JSON data
    try:
        return json.loads(json_data)
    except json.JSONDecodeError:
        return None


class InterviewQuestions(BaseModel):
    questions: List[str] = Field(
        description="List of interview questions based on topic and programming language"
    )


class AnswerEvaluation(BaseModel):
    score: float = Field(
        description="Score based on the number of questions answered correctly"
    )
    correct_answers: List[dict] = Field(
        description="Model generated answers for the questions"
    )


class GoogleAssistant:
    def __init__(self, model_name: str):
        self.model = VertexAI(model_name="gemini-pro")
        self.question_parser = PydanticOutputParser(pydantic_object=InterviewQuestions)
        self.answer_parser = PydanticOutputParser(pydantic_object=AnswerEvaluation)
        self.question_prompt = PromptTemplate(
            template="Generate 10 {level} level interview questions based on the topic {topic} and the programming language {language}.\n{format_instructions}\n",
            input_variables=["topic", "language", "level"],
            partial_variables={
                "format_instructions": self.question_parser.get_format_instructions()
            },
        )
        self.answer_prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    "Evaluate the following answers to the given questions and provide a score as a percentage out of 100. "
                    "For each question that has been answered incorrectly, provide the question along with the correct answer.\n"
                    "{format_instructions}\nQuestions: {questions}\nAnswers: {answers}\n"
                    "Please return the results in JSON format, including only the questions with incorrect answers and the correct answers.And make sure to properly check and validate all the answers"
                )
            ],
            input_variables=["questions", "answers"],
            partial_variables={
                "format_instructions": self.answer_parser.get_format_instructions()
            },
        )

    def evaluate_answers(self, qa_pairs: List[dict]) -> dict:
        questions = [qa_pair.question for qa_pair in qa_pairs]
        answers = [qa_pair.answer for qa_pair in qa_pairs]
        prompt_and_model = self.answer_prompt | self.model
        output = prompt_and_model.invoke({"questions": questions, "answers": answers})
        output = extract_json(output)
        output_dict = output

        # Transform correct_answers into the expected format
        correct_answers_dicts = [
            {
                "question": correct_answer_dict["question"],
                "answer": correct_answer_dict.get("answer")
                or correct_answer_dict.get("correct_answer"),
            }
            for correct_answer_dict in output_dict["correct_answers"]
        ]

        # Return the transformed output
        return {"score": output_dict["score"], "correct_answers": correct_answers_dicts}

    def generate_questions(self, topic: str, languages: List[str], level: str) -> dict:
        all_questions = []
        for language in languages:
            prompt_and_model = self.question_prompt | self.model
            output = prompt_and_model.invoke(
                {"topic": topic, "language": language, "level": level}
            )
            questions = self.question_parser.invoke(output)
            all_questions.extend(questions.questions)
        return {"questions": all_questions}


# assistant = GoogleAssistant(model_name="gpt")
# topic = "data science"
# languages = ["Python", "SQL", "Maths"]
# level = "beginner"
# questions_dict = assistant.generate_questions(topic, languages, level)
# print(json.dumps(questions_dict, indent=4))

# qa_pairs_payload = {
#     "qa_pairs": [
#         {
#             "question": "What is Data Science and how does it differ from traditional statistics?",
#             "answer": "Data Science is a multidisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It differs from traditional statistics in that it incorporates techniques and theories from broader fields like mathematics, statistics, information science, and computer science. Traditional statistics, on the other hand, is more focused on data analysis and interpretation in isolation.",
#         },
#         {
#             "question": "Can you briefly explain the data science lifecycle?",
#             "answer": "The data science lifecycle typically includes the following steps: 1) Business Understanding - defining the problem and desired outcomes. 2) Data Mining - gathering the required data. 3) Data Cleaning - removing inconsistencies and inaccuracies in the data. 4) Data Exploration - using statistical techniques to identify patterns or anomalies in the data. 5) Feature Engineering - modifying or creating new variables to better represent the underlying data patterns. 6) Predictive Modeling - creating a statistical model for prediction. 7) Data Visualization - visualizing the data and model outcomes. 8) Deployment - implementing the model into production. 9) Monitoring - tracking the model's performance over time.",
#         },
#         {
#             "question": "What are some common libraries in Python used for data science and what are their uses?",
#             "answer": "Some common Python libraries used in data science include: 1) NumPy - used for numerical computations and handling arrays. 2) Pandas - used for data manipulation and analysis. 3) Matplotlib - used for data visualization. 4) Scikit-learn - used for machine learning and statistical modeling. 5) TensorFlow and PyTorch - used for deep learning.",
#         },
#         {
#             "question": "What is a pandas DataFrame in Python and how is it used in data science?",
#             "answer": "I have no idea.",
#         },
#         {
#             "question": "Can you describe what 'data cleaning' is and why it's important in data science?",
#             "answer": "Data cleaning is way to hack python to produce desired output",
#         },
#     ]
# }

# from models.answer import AnswerEvaluationRequest, QuestionAnswerPair
# qa_pairs = [QuestionAnswerPair(**qa_pair) for qa_pair in qa_pairs_payload["qa_pairs"]]
# evaluation_request = AnswerEvaluationRequest(qa_pairs=qa_pairs)
# evaluation_dict = assistant.evaluate_answers(evaluation_request.qa_pairs)
# print(json.dumps(evaluation_dict, indent=4))
