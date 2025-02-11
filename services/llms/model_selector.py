from enum import Enum
from .anthropic_provider import AnthropicProvider
from .vertex_provider import VertexProvider
from .aws_provider import AWSProvider
from services.config.config_model import ModelProviderParams, ModelProvider

class ModelSelector:

    def __init__(self, model_parameters: ModelProviderParams):
        model_provider = model_parameters.provider
        self.model_id = model_parameters.llm_model_id
        self.temperature = model_parameters.temperature
        self.max_tokens = model_parameters.max_tokens
        self.provider_args = model_parameters.provider_args
        self.model = None
        if model_provider == ModelProvider.AWS:
            print(f"Loading AWS Provider: {self.model_id}")
            aws_provider = AWSProvider(self.model_id, self.temperature, self.max_tokens, **self.provider_args)
            self.model = aws_provider.retrieve_aws_chat_converse()
        elif model_provider == ModelProvider.GOOGLE:
            print(f"Loading GOOGLE Provider: {self.model_id}")
            # TODO pass max tokens
            vertex_provider = VertexProvider(self.model_id, self.temperature, **self.provider_args)
            self.model = vertex_provider.retrieve_vertex_chat()
        elif model_provider == ModelProvider.ANTHROPIC:
            # TODO pass max tokens
            print(f"Loading ANTHROPIC Provider: {self.model_id}")
            anthropic_provider = AnthropicProvider(self.model_id, self.temperature)
            self.model = anthropic_provider.retrieve_anthropic_chat()
        else:
            raise Exception("could not identify the model provider")