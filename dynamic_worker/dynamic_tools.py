import json

from langchain_core.tools import tool, StructuredTool
from functools import partial
from .utilities import model_from_schema, request_service

# THIS FUNCTION COULD BE STATIC
def call_api(endpoint, config, **args):
    # delegate the dynamic model to pass the args
    # At the end of the day this code is going to call APIs
    try:
        print("endpoint: ", endpoint)
        print("config: ", config)
        print("arguments function: ", args)
        request_body = {
            "data": {
                **args
            }
        }
        request_headers = config["headers"]
        response = request_service(endpoint, request_body, request_headers)
        print(response)
        return response
    except Exception as e:
        print(f"Catch exception: {e}")


class DynamicTools:

    def __init__(self, tool_name, tool_description, tool_json_schema, tool_api_endpoint, tool_api_config ):
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_model = model_from_schema(tool_json_schema)
        self.tool_api_endpoint = tool_api_endpoint
        self.tool_api_config = tool_api_config


    # TODO check this thing
    def gen_tool(self):
        tool_function = partial(call_api, self.tool_api_endpoint, self.tool_api_config)
        custom_tool = StructuredTool.from_function(
            func=tool_function,
            name=self.tool_name,
            description=self.tool_description,
            args_schema=self.tool_model,
            return_direct=True
        )
        return custom_tool