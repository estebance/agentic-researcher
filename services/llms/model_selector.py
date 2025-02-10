from enum import Enum

from pyasn1_modules.rfc1905 import max_bindings

from .anthropic_provider import AnthropicProvider
from .vertex_provider import VertexProvider
from .aws_provider import AWSProvider

class ModelProvider(Enum):
    AWS = 'AWS'
    GOOGLE = 'GOOGLE'
    ANTHROPIC = 'ANTHROPIC'


class ModelSelector:

    def __init__(self, provider, model_id, temperature, max_tokens, **provider_args):
        self.provider = provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_args = provider_args

    def get_model(self):
        model = None
        model_provider = self.provider
        if model_provider == ModelProvider.AWS:
            print(f"Loading AWS Provider: {self.model_id}")
            aws_provider = AWSProvider(self.model_id, self.temperature, self.max_tokens, **self.provider_args)
            model = aws_provider.retrieve_aws_chat_converse()
        elif model_provider == ModelProvider.GOOGLE:
            print(f"Loading GOOGLE Provider: {self.model_id}")
            # TODO pass max tokens
            vertex_provider = VertexProvider(self.model_id, self.temperature, **self.provider_args)
            model = vertex_provider.retrieve_vertex_chat()
        elif model_provider == ModelProvider.ANTHROPIC:
            # TODO pass max tokens
            print(f"Loading ANTHROPIC Provider: {self.model_id}")
            anthropic_provider = AnthropicProvider(self.model_id, self.temperature)
            model = anthropic_provider.retrieve_anthropic_chat()
        else:
            raise Exception("could not identify the model provider")
        return model