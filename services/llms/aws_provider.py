from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_core.callbacks import StdOutCallbackHandler

class AWSProvider:

    def __init__(self, model_id: str, temperature: float, max_tokens: int, model_region: str = "us-west-2"):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stdout_callback_handler = StdOutCallbackHandler()
        self.model_region = model_region


    def retrieve_aws_chat(self):
        # TODO bedrock chat
        return ChatBedrock(
            model_id=self.model_id,
            model_kwargs={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            callbacks=[self.stdout_callback_handler]
        )

    def retrieve_aws_chat_converse(self):
        # TODO bedrock chat converse
        return ChatBedrockConverse(
            model_id=self.model_id,
            temperature=self.temperature,
            callbacks=[self.stdout_callback_handler],
            max_tokens=self.max_tokens,
            region_name=self.model_region
        )