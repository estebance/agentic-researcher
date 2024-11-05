from enum import Enum
from anthropic_provider import AnthropicProvider
from vertex_provider import VertexProvider

class ModelProvider(Enum):
    AWS = 'AWS'
    GOOGLE = 'GOOGLE'
    ANTHROPIC = 'ANTHROPIC'


class ModelSelector:

    def __init__(self, provider, model_id, **provider_args):
        self.provider = provider
        self.model_id = model_id
        self.provider_args = provider_args


    def get_model(self):
        model = None
        model_provider = ModelProvider[self.provider]

        if model_provider == ModelProvider.AWS:
            model = AwsProvider(self.model_id, *self.provider_args)
        elif model_provider == ModelProvider.GOOGLE:
            model = VertexProvider(self.model_id, **self.provider_args)
            # return model
        elif model_provider == ModelProvider.ANTHROPIC:
            anthropic_provider = AnthropicProvider(self.model_id)

        else:
            raise Exception("could not identify the model provider")





