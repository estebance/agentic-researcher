from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from state import  State
from tools import tools

load_dotenv()

llm_model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
llm_model = llm_model.bind_tools(tools)

def chatbot(state: State):
    # print(state["pregunta"])
    response = llm_model.invoke(state["messages"])
    print(response)
    return {
        "messages": [response]
    }


graph_builder = StateGraph(State)
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

# add first and end nodes (edges)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge("chatbot", END)

# compiled graph
graph = graph_builder.compile()

message_inputs = [HumanMessage(content="Que planes de turismo tienes disponibles, mi nombre es Esteban, tengo 33 años y my id es 1?")]
graph.invoke({"messages": message_inputs})