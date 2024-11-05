import os
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import StdOutCallbackHandler

import json
class AnthropicProvider:

    def __init__(self, model_id, temperature=0):
        self.model_id = model_id
        self.stdout_callback_handler = StdOutCallbackHandler()
        self.temperature = temperature


    def retrieve_anthropic_chat(self):
        return ChatAnthropic(
            model=self.model_id,
            temperature=self.temperature
        )
