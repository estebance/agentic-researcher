from tabnanny import process_tokens

from dotenv import load_dotenv

load_dotenv()
import functools
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph, MessagesState
from IPython.display import Image, display
# from .services.redis_checkpointer.redis_saver import RedisSaver
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
# state

memory = MemorySaver()
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    formulated_question: str
    question: str

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
    }

def create_agent(llm):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    Given a chat history and the latest user question \n\
                    which might reference context in the chat history, formulate a standalone question\n\
                    which can be understood without the chat history. Do NOT answer the question,\n\
                    just reformulate it if needed and otherwise return it as is and proceed. \n\
                    Do NOT provide explanations just return the question""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt | llm

rewriter_agent = create_agent(
    model,
)
rewriter_node = functools.partial(agent_node, agent=rewriter_agent, name="Rewriter")

workflow = StateGraph(State)
workflow.add_node("Rewriter", rewriter_node)

workflow.add_edge(START, "Rewriter")
workflow.add_edge( "Rewriter", END)
graph = workflow.compile(checkpointer=memory)


def check_state(llm_graph, config, user_input):
    current_state = llm_graph.get_state(config).values
    fulfill_message = True
    if "messages" in current_state:
        last_node = current_state["messages"][-1]
        print(last_node)
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
    # print(f"Final state: {final_state}")
    response = final_state["messages"][-1]
    if isinstance(response, AIMessage) and response.tool_calls:
        print(response.tool_calls)
        tool_call = response.tool_calls[0]
        print(tool_call)
        if tool_call["name"] == "AskHuman":
            response = tool_call["args"]["approval"]
    else:
        response = response.content
    return response

def generate_graph():
    try:
        image = Image(graph.get_graph(xray=True).draw_mermaid_png())
        with open("rewriter.png", "wb") as fout:
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
    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "user_id": user_id,
            # Checkpoints ar2e accessed by thread_id
            "thread_id": thread_id,
        }
    }
    message_inputs = [HumanMessage(content=human_message)]
    final_state = graph.invoke(
        {"messages": message_inputs}, config
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

# generate_graph()
# process_request('restebance@gmail.com', '1234', human_message="que es la cop 16?")
# process_request('restebance@gmail.com', '1234', human_message="que eventos tiene?")






