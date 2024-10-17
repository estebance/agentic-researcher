import json
from pydantic import BaseModel, ValidationError
PARAMETERS_FILE = "params.json"


class ParametrizationAgent(BaseModel):
    provider: str
    model_id: str
    knowledge_base_id: str


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