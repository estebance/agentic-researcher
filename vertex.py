import os
import vertexai
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_google_vertexai import ChatVertexAI
from langchain_core.callbacks import StdOutCallbackHandler
from google.oauth2 import service_account


GCP_MODEL_ID = os.environ.get('GCP_MODEL_ID')
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
GCP_PROJECT_REGION = os.environ.get('GCP_PROJECT_REGION')

import json

with open('sa.json', 'r') as file:
    vertex_sa = json.load(file)

credentials = service_account.Credentials.from_service_account_info(vertex_sa, scopes=["https://www.googleapis.com/auth/cloud-platform"])
vertexai.init(project=GCP_MODEL_ID, location=GCP_PROJECT_REGION, credentials=credentials)
stdout_callback_handler = StdOutCallbackHandler()

def load_vertex_model_anthropic():
    model = ChatAnthropicVertex(
        model_name=GCP_MODEL_ID,
        project=GCP_PROJECT_ID,
        location=GCP_PROJECT_REGION,
        credentials=credentials,
        callbacks=[stdout_callback_handler]
    )
    return model

def load_vertex_model_gemini():
    model = ChatVertexAI(
        model_name=GCP_MODEL_ID,
        project=GCP_PROJECT_ID,
        location=GCP_PROJECT_REGION,
        credentials=credentials,
        callbacks=[stdout_callback_handler]
    )
    return model