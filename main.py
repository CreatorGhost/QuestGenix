from dotenv import load_dotenv
import firebase_admin
from firebase_admin import auth,exceptions
from fastapi import FastAPI, Body, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware

from models.question import QuestionGenerationRequest, QuestionGenerationResponse
from models.answer import AnswerEvaluationRequest, AnswerEvaluationResponse
from helper.interview_assistant import InterviewAssistant
import logging
from fastapi.encoders import jsonable_encoder
load_dotenv()  




origins = ["*"]

firebase_admin.initialize_app()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/users/me/")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    try:
        # Verify the ID token while checking if the token is revoked by
        # passing check_revoked=True.
        decoded_token = auth.verify_id_token(token, check_revoked=True)
        # Token is valid and not revoked.
        uid = decoded_token['uid']
        print(f"Token is valid and not revoked. User ID: {uid}")
    except ValueError:
        # Token was not a valid ID token.
        print("Token was not a valid ID token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        # Token was revoked or expired
        print("Token was revoked or expired.")
        print("The eror is ",e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"uid": uid}



async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        # Verify the ID token while checking if the token is revoked by
        # passing check_revoked=True.
        decoded_token = auth.verify_id_token(token, check_revoked=True)
        # Token is valid and not revoked.
        uid = decoded_token['uid']
        return uid
    except ValueError:
        # Token was not a valid ID token.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception:
        # Token was revoked or expired
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/protected/")
async def read_protected_route(current_user: str = Depends(get_current_user)):
    return {"message": "This is a protected route"}


assistant = InterviewAssistant(model_name="gpt-4")


@app.post("/generate_questions/", response_model=QuestionGenerationResponse)
async def generate_questions(request: QuestionGenerationRequest):
    try:
        request_data = jsonable_encoder(request)
        logging.info(f"Generating questions with payload: {request_data}")
        questions_dict = assistant.generate_questions(request.topic, request.languages, request.level)
        logging.info(f"Questions generated: {questions_dict}")
        return questions_dict
    except Exception as e:
        logging.error(f"Error generating questions with payload {request_data}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating questions."
        )

@app.post("/evaluate_answers/", response_model=AnswerEvaluationResponse)
async def evaluate_answers(request: AnswerEvaluationRequest):
    try:
        request_data = jsonable_encoder(request)
        logging.info(f"Evaluating answers with payload: {request_data}")
        evaluation_dict = assistant.evaluate_answers(request.qa_pairs)
        logging.info(f"Answers evaluated: {evaluation_dict}")
        return evaluation_dict
    except Exception as e:
        logging.error(f"Error evaluating answers with payload {request_data}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while evaluating answers."
        )