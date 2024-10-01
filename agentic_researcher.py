from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from tools import retrieve_with_vectorstore, retrieve_with_web_search
from assistant import Assistant
from state import State
from utilities import create_tool_node_with_fallback, _print_event

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Call the model
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an experienced AI researcher
            """
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())
# Partial such a good way to index information
# Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation
tools = [retrieve_with_vectorstore, retrieve_with_web_search]
#
assistant_runnable = primary_assistant_prompt | model.bind_tools(tools)


builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# from IPython.display import Image, display
# try:
#     display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass
questions = [
    "Quiero buscar información sobre como migrar a Canada"
]

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "user_id": "restebance",
        # Checkpoints are accessed by thread_id
        "thread_id": "43",
    }
}


_printed = set()
for question in questions:
    events = graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)

