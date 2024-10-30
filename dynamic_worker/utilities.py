import argparse
from pydantic import create_model
import requests

# Example JSON schema
# json_schema = {
#     "title": "Example Model",
#     "type": "object",
#     "properties": {
#         "id": {
#             "type": "integer",
#         },
#         "name": {
#             "type": "string",
#         },
#         "age": {
#             "type": "integer",
#             "default": 30
#         },
#     },
#     "required": ["id", "name"]
# }

# Function to dynamically create a pydantic model from json schema
def model_from_schema(schema_dict):
    fields = {}
    print("schema dict:", schema_dict['title'])
    function_arguments = argparse.ArgumentParser()
    for field_name, field_info in schema_dict['properties'].items():
        field_type = {'string': str, 'integer': int, 'number': float, 'boolean': bool, 'object': object}.get(field_info['type'], None)
        default_value = field_info.get("default", Ellipsis if field_name in schema_dict.get('required', []) else None)
        fields[field_name] = (field_type, default_value)
        # add arguments
        function_arguments.add_argument(field_name, type=field_type)
    model = create_model(schema_dict['title'], **fields)
    return model


def request_service(endpoint, request_body, request_headers):
    try:
        response = requests.post(endpoint, json=request_body, headers=request_headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")