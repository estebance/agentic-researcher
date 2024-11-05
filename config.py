import json
from pydantic import BaseModel, ValidationError
from typing import Optional
from services.llms.model_selector import ModelProvider

PARAMETERS_FILE = "params.json"

class CheckpointerAuthParams(BaseModel):
    username: str
    password: str
    ssl: bool

class CheckpointerParams(BaseModel):
    endpoint: str
    port: int
    db_number: int
    auth_params: Optional[CheckpointerAuthParams] = None

class WebRetrieverParams(BaseModel):
    enabled: bool
    urls: list[str]
    is_advanced_search: bool
    max_number_of_resources: int

class KdbRetrieverParams(BaseModel):
    kdb_id: str
    kdb_max_number_of_results: int
    kdb_region: str

class ToolsParams(BaseModel):
    name: str
    description: str
    tool_schema: dict
    endpoint: str
    endpoint_config: dict

class WokerParams(BaseModel):
    id: str
    name: str
    task: str
    tools: list[ToolsParams]

class AssistantWorkerParams(BaseModel):
    id: str
    task: str

class ResearcherWorkerParams(BaseModel):
    id: str
    task: str
    enabled: bool
    web_retriever: WebRetrieverParams
    kdb_retriever_params: KdbRetrieverParams

class ModelProviderParams(BaseModel):
    name: ModelProvider
    model_id: str
    provider_args: Optional[dict] = None

class ParametrizationAgent(BaseModel):
    provider: str
    llm_model_id: str
    model_provider: Optional[ModelProviderParams] = None
    checkpointer: CheckpointerParams
    workers: list[WokerParams]
    assistant_worker: AssistantWorkerParams
    researcher_worker: ResearcherWorkerParams

def validate_parametrization_file(json_data):
    try:
        parametrization = ParametrizationAgent(**json_data)
        return parametrization
    except ValidationError as e:
        print("the provided config format is not valid: ", e)
        raise e


def retrieve_parameters():
    parameters = None
    with open(PARAMETERS_FILE, 'r') as file:
        data = json.load(file)
        parameters = validate_parametrization_file(data)
    return parameters

if __name__ == "__main__":
    retrieve_parameters()