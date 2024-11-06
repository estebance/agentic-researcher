import os
import vertexai
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_google_vertexai import ChatVertexAI
from langchain_core.callbacks import StdOutCallbackHandler
from google.oauth2 import service_account

import json


class VertexProvider:

    def __init__(self, model_id, temperature, **provider_args):
        self.model_id = model_id
        self.temperature = temperature
        self.project_id = provider_args["project_id"]
        self.location = provider_args["location"]
        self.sa = json.loads(provider_args["sa"])
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
            model_name=self.model_id,
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
            callbacks=[self.stdout_callback_handler]
        )


    # TODO retrieve vertex chat
    def retrieve_vertex_chat(self):
        model = None
        if self.model_id.startswith("gemini"):
            model = self.load_vertex_model_gemini()
        elif self.model_id.startswith("claude"):
            model = self.load_vexter_model_anthropic()
        else:
            raise Exception(f"model name not identified: {self.model_id}")
        return model
