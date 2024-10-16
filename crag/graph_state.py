from typing import List, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        messages: List of messages
    """
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    generation: str
    web_search: str
    documents: List[str]