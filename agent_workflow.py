import operator
from config import retrieve_parameters
from langchain_aws import ChatBedrock
from agent.agent import AgentSupervisor
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage
from services.redis_checkpointer.redis_saver import RedisSaver
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import BaseMessage
from typing import Sequence


class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    response: str
    know_reply: bool


config_parameters = retrieve_parameters()
model = ChatBedrock(model_id=config_parameters.llm_model_id, temperature=0)

agent_supervisor = AgentSupervisor(model, name='Agent Smith', role='Software Developer', features="Find bugs in agents\n Like programming and Coffee", language="es")
# role descriptor




agent_graph = StateGraph(AgentState)
agent_graph.add_node("AgentSupervisor", agent_supervisor.agent_node)

# Define the control flow
agent_graph.add_edge(START, "AgentSupervisor")
agent_graph.add_edge("AgentSupervisor", END)

def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results

with RedisSaver.from_conn_info(host="localhost", port=6379, db=1) as checkpointer:
    chain = agent_graph.compile(
        checkpointer=checkpointer
    )
    # The following functions interoperate between the top level graph state
    # and the state of the research sub-graph
    # this makes it so that the states of each graph don't get intermixed
    research_chain = enter_chain | chain

for s in research_chain.stream(
    "Cuentame sobre ti?",
    {
        "recursion_limit": 150,
        "user_id": "1",
        "thread_id": "1"
    },
):
    print("stream: ", s.keys())
    print("stream: ", s)