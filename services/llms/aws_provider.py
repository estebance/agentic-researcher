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

class AWSProvider:

    def __init__(self, model_id):
        self.model_id = model_id
        self.stdout_callback_handler = StdOutCallbackHandler()


    def load_model_anthropic(self):
        # TODO bedrock chat
        return ChatAnthropicVertex(
            model_name=self.model_id,
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
            callbacks=[self.stdout_callback_handler]
        )
