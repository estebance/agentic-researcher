from tabnanny import process_tokens

from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.postgres import PostgresSaver
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from datetime import datetime
from IPython.display import Image, display
from tools import web_search_tool, AskHuman
from assistant import Assistant
from state import State
from utilities import create_tool_node_with_fallback, _print_event
# from services.checkpointer import sync_checkpointer, retrieve_sync_connection_checkpointer, retrieve_async_connection_checkpointer
from services.redis_checkpointer.redis_checkpointer import retrieve_sync_connection_checkpointer
from services.redis_checkpointer.redis_saver import RedisSaver
from langchain_core.messages import ToolMessage

# fake node
def ask_human(state):
    pass

#
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
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
assistant_runnable = primary_assistant_prompt | model.bind_tools(tools + [AskHuman])
builder = StateGraph(State)
# Define nodes: these do the work
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))
builder.add_node("ask_human", ask_human)
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
        "ask_human": "ask_human",
        # Otherwise we finish.
        "end": END,
    },
)
builder.add_edge("ask_human", "assistant")

# builder.add_edge("assistant", "tools")
# builder.add_conditional_edges(
#     "assistant",
#     tools_condition,
# )
builder.add_edge("tools", "assistant")


def check_state(llm_graph, config, user_input):
    current_state = llm_graph.get_state(config).values
    print(current_state)
    fulfill_message = True
    if "messages" in current_state:
        last_node = current_state["messages"][-1]
        if isinstance(last_node, AIMessage) and last_node.tool_calls:
            tool_call = last_node.tool_calls[-1]
            if tool_call and tool_call["name"] == "AskHuman":
                tool_call_id =tool_call["id"]
                tool_message = [
                    {"tool_call_id": tool_call_id, "type": "tool", "content": user_input}
                ]
                llm_graph.update_state(config, {"messages": tool_message}, as_node="ask_human")
                fulfill_message = False
    return llm_graph, fulfill_message

def retrieve_response(final_state):
    print(f"Final state: {final_state}")
    response = final_state["messages"][-1]
    if isinstance(response, AIMessage) and response.tool_calls:
        print(response.tool_calls)
        tool_call = response.tool_calls[0]
        print(tool_call)
        if tool_call["name"] == "AskHuman":
            response = tool_call["args"]["approval"]
    else:
        response = response.content
    print(f"Final response: {response}")
    return response

def generate_graph():
    with RedisSaver.from_conn_info(host="localhost", port=6379, db=0) as checkpointer:
        llm_graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["ask_human"],
        )
        try:
            image = Image(llm_graph.get_graph(xray=True).draw_mermaid_png())
            with open("net_image1.png", "wb") as fout:
                fout.write(image.data)
            # display(image)
        except Exception:
            # This requires some extra dependencies and is optional
            pass
# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
# checkpointer =  retrieve_sync_connection_checkpointer()
def process_request(user_id, thread_id, human_message):
    response = None
    with RedisSaver.from_conn_info(host="localhost", port=6379, db=0) as checkpointer:
        llm_graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["ask_human"],
        )
        config = {
            "configurable": {
                # The passenger_id is used in our flight tools to
                # fetch the user's flight information
                "user_id": "restebance",
                # Checkpoints are accessed by thread_id
                "thread_id": "21",
            }
        }
        # if the last intaction was a request to the user avoid insertion of new messages
        llm_graph, fulfill_message = check_state(llm_graph, config, human_message)
        final_state = None
        if fulfill_message:
            message_inputs = [HumanMessage(content=human_message)]
            final_state = llm_graph.invoke(
                {"messages": message_inputs}, config
            )
        else:
            final_state = llm_graph.invoke(
                None, config
            )
        response = retrieve_response(final_state)
        print(f"Final response: ", response)
        return response
# check graph
# try:
#     image = Image(llm_graph.get_graph(xray=True).draw_mermaid_png())
#     with open("net_image1.png", "wb") as fout:
#         fout.write(image.data)
#     # display(image)
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass
# process_request('restebance@gmail.com', '1234', human_message="a que hora es el evento Rangers at the heart of the 30x30 de la COP16?")
# generate_graph()
# process_request('restebance@gmail.com', '1234', human_message="No paila")






