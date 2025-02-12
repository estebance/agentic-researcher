from fastapi import FastAPI, Request
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
from supervisor_workflow import SupervisorWorkflow

load_dotenv()

app = FastAPI()

supervised_workflow = SupervisorWorkflow()
supervised_chain = supervised_workflow.gen_chain()

@app.get("/")
async def default():
    return {
        "data": {
            "msg": "success"
        }
    }

# # TODO manage request body
# class SupervisorData(BaseModel):
#     message: str
#     user_id: str
#     thread_id: str
#
# # TODO provide the thread_id and the user_id
# @app.post("/supervisor")
# async def supervisor(supervisor_request: SupervisorData):
#     request_body = supervisor_request.model_dump()
#     message = request_body["message"]
#     reply = supervised_chain.invoke(
#         message,
#         {
#             "recursion_limit": 150,
#             "user_id": "restebance@gmail.com",
#             "thread_id": "16"
#         },
#     )
#     print(reply['response'])
#     return {
#         "data": {
#             "message": reply['response']
#         }
#     }