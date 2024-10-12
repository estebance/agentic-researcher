# state as a typed dictionary containing an append-only list of messages.
# These messages form the chat history, which is all the state our simple assistant needs.

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
