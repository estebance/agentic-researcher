import json
from pydantic import BaseModel, ValidationError
PARAMETERS_FILE = "params.json"


class CheckpointerParams(BaseModel):
    endpoint: str
    port: int
    db_number: int

class WebRetrieverParams(BaseModel):
    urls: list[str]
    is_advanced_search: bool
    max_number_of_resources: int

class KdbRetrieverParams(BaseModel):
    kdb_id: str
    kdb_max_number_of_results: int
    kdb_region: str

class ParametrizationAgent(BaseModel):
    provider: str
    llm_model_id: str
    kdb_retriever_params: KdbRetrieverParams
    web_retriever: WebRetrieverParams
    checkpointer: CheckpointerParams


def validate_parametrization_file(json_data):
    try:
        parametrization = ParametrizationAgent(**json_data)
        print(parametrization)
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