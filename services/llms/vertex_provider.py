import os
import vertexai
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_google_vertexai import ChatVertexAI
from langchain_core.callbacks import StdOutCallbackHandler
from google.oauth2 import service_account

import json

GCP_MODEL_ID = os.environ.get('GCP_MODEL_ID')
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
GCP_PROJECT_REGION = os.environ.get('GCP_PROJECT_REGION')

class VertexProvider:

    def __init__(self, model_id, temperature, **vertex_params ):
        self.model_id = model_id
        self.temperature = temperature
        self.project_id = vertex_params["project_id"]
        self.location = vertex_params["location"]
        self.sa = vertex_params["sa"]
        self.project_id = self.project_id
        self.location = self.location
        self.credentials = service_account.Credentials.from_service_account_info(self.sa, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        vertexai.init(project=model_id, location=self.location, credentials=self.credentials)
        self.stdout_callback_handler = StdOutCallbackHandler()
        self.temperature = temperature


    def load_vexter_model_anthropic(self):
        return ChatAnthropicVertex(
            model_name=self.model_id,
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
            callbacks=[self.stdout_callback_handler]
        )


    def load_vertex_model_gemini(self):
        return ChatVertexAI(
            model_name=GCP_MODEL_ID,
            project=GCP_PROJECT_ID,
            location=GCP_PROJECT_REGION,
            credentials=self.credentials,
            callbacks=[self.stdout_callback_handler]
        )

# with open('sa.json', 'r') as file:
#     vertex_sa = json.load(file)
#
# credentials = service_account.Credentials.from_service_account_info(vertex_sa, scopes=["https://www.googleapis.com/auth/cloud-platform"])
# vertexai.init(project=GCP_MODEL_ID, location=GCP_PROJECT_REGION, credentials=credentials)

