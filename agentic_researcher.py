from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.postgres import PostgresSaver
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from datetime import datetime
from IPython.display import Image, display
from tools import web_search_tool
from assistant import Assistant
from state import State
from utilities import create_tool_node_with_fallback, _print_event
# from services.checkpointer import sync_checkpointer, retrieve_sync_connection_checkpointer, retrieve_async_connection_checkpointer
from services.redis_checkpointer.redis_checkpointer import retrieve_sync_connection_checkpointer
from services.redis_checkpointer.redis_saver import RedisSaver
from langchain_core.messages import ToolMessage


#
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    print(last_message)
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Call the model
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

# \n\nCurrent user:\n<User>\n{user_info}\n</User>
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an experienced AI researcher expert in topics culture, history, enviromental stuff.
                Use the provided tools to search for information about Colombian culture, history and the COP16 event
                If a search comes up empty, expand your search before giving up.
                Your responses include no more than three sentences.
                "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            """
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())
# Partial such a good way to index information
# Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation
tools = [web_search_tool]
#
assistant_runnable = primary_assistant_prompt | model.bind_tools(tools)
builder = StateGraph(State)
# Define nodes: these do the work
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "assistant",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END,
    },
)

# builder.add_edge("assistant", "tools")
# builder.add_conditional_edges(
#     "assistant",
#     tools_condition,
# )
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
# checkpointer =  retrieve_sync_connection_checkpointer()
with RedisSaver.from_conn_info(host="localhost", port=6379, db=0) as checkpointer:
    llm_graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )

    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "user_id": "restebance",
            # Checkpoints are accessed by thread_id
            "thread_id": "12",
        }
    }
    message_inputs = [HumanMessage(content="a que hora es el evento Rangers at the heart of the 30x30 de la COP16?")]
    _printed = set()
    events = llm_graph.stream(
        {"messages": message_inputs}, config, stream_mode="values"
    )
    # events = llm_graph.stream(
    #     {"messages": ("user", "Si dale")}, config, stream_mode="values"
    # )
    for event in events:
        _print_event(event, _printed)

    user_input = input(
        "Do you approve of the above actions? Type 'y' to continue;"
        " otherwise, explain your requested changed.\n\n"
    )

    events_b =  llm_graph.stream(None, config, stream_mode="values")
    for event_b in events_b:
        _print_event(event_b, _printed)

    # result = llm_graph.invoke(
    #     None,
    #     config,
    # )
    #
    # # snapshot = llm_graph.get_state(config)
    # # while snapshot.next:
    #     # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
    #     # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
    #     # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
    #
    #     if user_input.strip() == "y":
    #         # Just continue
    #
    #         print(result)
    #     else:
    #         # Satisfy the tool invocation by
    #         # providing instructions on the requested changes / change of mind
    #         result = llm_graph.invoke(
    #             {
    #                 "messages": [
    #                     ToolMessage(
    #                         tool_call_id=event["messages"][-1].tool_calls[0]["id"],
    #                         content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
    #                     )
    #                 ]
    #             },
    #             config,
    #         )
    #     snapshot = llm_graph.get_state(config)
    #     print(snapshot)
# check graph
try:
    image = Image(llm_graph.get_graph(xray=True).draw_mermaid_png())
    with open("net_image1.png", "wb") as fout:
        fout.write(image.data)
    # display(image)
except Exception:
    # This requires some extra dependencies and is optional
    pass






