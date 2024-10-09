from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from agentic_researcher import process_request

load_dotenv()

app = FastAPI()


class InteractionData(BaseModel):
    message: str
    user_id: str
    thread_id: str


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/interact")
async def interact(interaction_request: InteractionData):
    request_body = interaction_request.model_dump()
    user_id = request_body["user_id"]
    thread_id = request_body["thread_id"]
    message = request_body["message"]
    _printed = set()
    response = process_request(user_id, thread_id, message)

    return {"data": {
        "message": response
    }}