from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from agentic_researcher import llm_graph
from utilities import _print_event

load_dotenv()

app = FastAPI()


class InteractionData(BaseModel):
    question: str
    user_id: str
    thread_id: str


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/interact")
async def interact(interaction_request: InteractionData):
    request_body = interaction_request.model_dump()
    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "user_id": request_body["user_id"],
            # Checkpoints are accessed by thread_id
            "thread_id": request_body["thread_id"],
        }
    }
    _printed = set()
    events = llm_graph.stream(
        {"messages": ("user", request_body["question"])}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
        if "__end__" in event:
            print("----")
            print(event)
            print("----")
    # _printed = set()
    # for question in questions:
    #     events = graph.stream(
    #         {"messages": ("user", question)}, config, stream_mode="values"
    #     )
    return {"message": "Hello World"}