import json
import os
from services.kdbs.kdb_retriever import KDBRetriever
from services.config.config import Config
from services.llms.model_selector import ModelSelector

class ConfigRetriever:

    def __init__(self, config_parameters_file_name: str):
        self.parametrization = None
        current_directory = os.path.dirname(__file__)
        config_parameters_file = os.path.join(current_directory, config_parameters_file_name)
        if not os.path.exists(config_parameters_file):
            raise Exception("no file found")
        with open(config_parameters_file, 'r') as file_content:
            data = json.load(file_content)
            config_local = Config(data)
            self.parametrization = config_local.parametrization
        model_params = self.parametrization.llm_model_provider
        kdb_params = self.parametrization.researcher_worker.bedrock_kdb_retriever_params
        checkpointer = self.parametrization.checkpointer
        model_selector = ModelSelector(
            model_params
        )
        self.model = model_selector.model
        kdb_retriever = KDBRetriever(kdb_params)
        self.retriever = kdb_retriever.retriever
        self.checkpointer = checkpointer

if __name__ == "__main__":
    config_retriever = ConfigRetriever('params.json')
