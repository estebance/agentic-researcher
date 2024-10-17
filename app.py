from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from crag_agent import process_request_crag

load_dotenv()

app = FastAPI()


class InteractionData(BaseModel):
    message: str
    user_id: str
    thread_id: str


@app.post("/ask")
async def crag(interaction_request: InteractionData):
    request_body = interaction_request.model_dump()
    user_id = request_body["user_id"]
    thread_id = request_body["thread_id"]
    message = request_body["message"]
    response = process_request_crag(user_id, thread_id, message)
    return {"data": {
        "message": response
    }}