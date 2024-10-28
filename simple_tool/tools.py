from langchain_core.tools import tool, StructuredTool
from typing import Annotated
from functools import partial
from pydantic import BaseModel, Field, create_model
from utilities import model_from_schema, request_service


@tool
def purchase_travel_plan(
    id_package: Annotated[str, "This is the id of the travel plan user would like to buy."],
):
    """Use this to purchase travel plans thru an api."""
    print(id_package)
    result_str = f"Successfully purchased the travel package: {id_package}"
    return result_str

#
# params = [{
#     "name": "TravelPlans",
#     "properties": [
#         {
#             "default": "hola",
#             "title": "user_type",
#             "type": "string"
#         },
#     ]
# }]
# print(properties)
# #
# fields = {}
# fields["user_type"] = (str, properties["default"])
#
# DynamicSchema = create_model(first_model, **fields)
# print(DynamicSchema)
# DynamicSchema(user_type="family")

# function
def fetch_travel_plans(user_type: str):
    """Use this to get information about travel plans"""
    return [
        {
            "name": "nice vacations",
            "desc": "nice vacations in Colombia",
            "id": 1
        },
        {
            "name": "mountain vacations",
            "desc": "Enjoy the sigth",
            "id": 2
        }
    ]

# model chema
json_schema = {
    "title": "TravelPlans",
    "type": "object",
    "properties": {
        "id": {
            "type": "integer",
        },
        "name": {
            "type": "string",
        },
        "age": {
            "type": "integer",
            "default": 30
        },
    },
    "required": ["id", "name"]
}


dynamic_model = model_from_schema(json_schema)

def my_dynamic_function(endpoint, **args):
    # delegate the dynamic model to pass the args
    # At the end of the day this code is going to call APIs
    print("endpoint: ", args)
    print("arguments function: ", args)
    endpoint = f"{endpoint}/plans"
    request_body = {
        "plans": ["hi"]
    }
    request_headers = {
        "Content-Type": "application/json"
    }
    response = request_service(endpoint, request_body, request_headers)
    print(response)
    return response

# TBD
def gen_tool(tool_name, tool_desc, tool_function, tool_function_model):
    custom_tool = StructuredTool.from_function(
        func=tool_function,
        name=tool_name,
        description=tool_desc,
        args_schema=tool_function_model,
        return_direct=True
    )
    return custom_tool

# defined as partial to add some configuration
dynamic_function_partial = partial(my_dynamic_function, 'http://localhost:8000')
# dynamic_function_partial(family_type="family")

fech_travel_plans_tool = gen_tool('fetch_travel_plans', 'Use this to get information about travel plans', dynamic_function_partial, dynamic_model)
tools = [purchase_travel_plan, fech_travel_plans_tool]
