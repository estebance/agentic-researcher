from fastapi import FastAPI, Request
import uvicorn
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



class PlansData(BaseModel):
    plans: list


@app.post("/plans")
async def plans(plans_request: PlansData):
    request_body = plans_request.model_dump()
    print(request_body)
    return {
        "data": {
            "plans": [
                {
                    "name": "nice vacations",
                    "desc": "nice vacations in Colombia",
                    "id": 1
                },
                {
                    "name": "mountain vacations",
                    "desc": "Enjoy the sigth",
                    "id": 2
                }
            ]
        }
    }

@app.post("/city-plans")


async def city_plans(request: Request):
    print(request)
    return {
        "data": {
            "activities": [
                {
                    "city": "Medellín",
                    "plans": [
                        {
                            "name": "have a good beer in TorreAlta",
                            "desc": "enjoy best beer in town",
                            "location": "Viva",
                            "date": "30/10/2024",
                            "hour": "2:00 PM",
                            "id": 1
                        },
                        {
                            "name": "have a good coffee",
                            "desc": "Enjoy a warn and nice cup of coffee in Laboratorio de Cafe",
                            "location": "Laureles",
                            "date": "30/10/2024",
                            "hour": "11:00 am",
                            "id": 2
                        }

                    ]
                }
            ]
        }
    }

@app.post("/buy-plan")
async def plans(plans_request: PlansData):
    request_body = plans_request.model_dump()
    print(request_body)
    return {
        "data": {
            "plan": {
                    "name": "nice vacations",
                    "id": 1
                }
        }
    }