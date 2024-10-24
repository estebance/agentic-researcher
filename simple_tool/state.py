from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages



class State(TypedDict):
    pregunta: str
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)