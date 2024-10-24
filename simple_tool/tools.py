from langchain_core.tools import tool, StructuredTool
from typing import Annotated
from functools import partial
from pydantic import BaseModel, Field, create_model
from utilities import model_from_schema
@tool
def purchase_travel_plan(
    id_package: Annotated[str, "This is the id of the travel plan user would like to buy."],
):
    """Use this to purchase travel plans thru an api."""
    print(id_package)
    result_str = f"Successfully purchased the travel package: {id_package}"
    return result_str

# schema

class TravelPlanInput(BaseModel):
    user_type: str = Field(description="user type: young, old, family")


#
params = [{
    "name": "TravelPlans",
    "properties": [
        {
            "default": "hola",
            "title": "user_type",
            "type": "string"
        },
    ]
}]

first_model = params[0]["name"]
properties = params[0]["properties"][0]


print(properties)
#
fields = {}
fields["user_type"] = (str, properties["default"])

DynamicSchema = create_model(first_model, **fields)
print(DynamicSchema)

DynamicSchema(user_type="family")

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
    "title": "Example Model",
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
def my_dynamic_function(*args):
    # delegate the dynamic model to pass the args
    # At the end of the day this code is going to call APIs
    print("arguments function: ", args)
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

fech_travel_plans_tool = gen_tool('fetch_travel_plans', 'Use this to get information about travel plans', my_dynamic_function, dynamic_model)

tools = [purchase_travel_plan, fech_travel_plans_tool]
