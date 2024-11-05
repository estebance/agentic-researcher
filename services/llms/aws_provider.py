from langchain_aws import ChatBedrock
from langchain_core.callbacks import StdOutCallbackHandler

class AWSProvider:

    def __init__(self, model_id, temperature):
        self.model_id = model_id
        self.temperature = temperature
        self.stdout_callback_handler = StdOutCallbackHandler()


    def retrieve_aws_chat(self):
        # TODO bedrock chat
        return ChatBedrock(
            model_id=self.model_id,
            temperature=self.temperature,
            callbacks=[self.stdout_callback_handler]
        )