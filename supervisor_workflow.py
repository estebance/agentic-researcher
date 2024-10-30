from typing import Sequence
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_core.messages import BaseMessage
from supervisor.supervisor import AgentSupervisor
from supervisor.nodes import SupervisorNodes
from simple_tool.app import process_request_vacations_planner_as_team
from dynamic_worker import DynamicWorker, DynamicTools
from langchain_anthropic import ChatAnthropic
from config import retrieve_parameters
from crag_agent import process_request_crag_as_team
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage
from services.redis_checkpointer.redis_saver import RedisSaver
import json
from pydantic import BaseModel, Field


config_parameters = retrieve_parameters()

# load a worker
# worker = { "worker_name": "city guide", "worker_task": "help users find relevant plans around the city"}
worker_model = ChatAnthropic(model=config_parameters.llm_model_id, temperature=0)
# worker_tools_info = {}
# worker_tools = []
# worker_id = "CityGuide"
# tool_json_schema = {
#     "title": "CityPlans",
#     "type": "object",
#     "properties": {
#         "city": {
#             "type": "string",
#         },
#     },
#     "required": ["city"]
# }

# for i, worker in enumerate(config_parameters.workers):
# print(f"Worker {i+1}: {worker}")
# print("Tools:", worker.tools)
worker = config_parameters.workers[0]
worker_tools = []
worker_id = worker.id
for tool in worker.tools:
    print(tool)
    dynamic_tool_def = DynamicTools(
        tool.name,
        tool.description,
        tool.tool_schema,
        tool.endpoint,
        tool.endpoint_config
    )
    dynamic_tool = dynamic_tool_def.gen_tool()
    worker_tools.append(dynamic_tool)
dynamic_worker = DynamicWorker(worker.id, worker.name, worker.task, worker_model, tools=worker_tools)



# reply
class ReplyToUser(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    response: str = Field(
        description="your reply to the user"
    )
    know_reply: bool = Field(
        description="You know the reply to the user"
    )

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], add_messages]
    know_reply: bool
    #
    response: str
    # Team
    team_members: List[str]
    # The 'next' field indicates where to route to next
    next: str
    summary: str
    cool_post: str




# from parameters we are going to retrieve
# custom agent name and custom agent role description
# pass the parameters to the simple_tool workflow as an agent to resolve a request
# The parameters are (tool name, tool_description, tool_schema)
print(config_parameters)
model = ChatAnthropic(model=config_parameters.llm_model_id, temperature=0)
supervisor_nodes = SupervisorNodes(model)

def decide_to_reply(state):
    if state["know_reply"]:
        return "FINISH"
    else:
        return "supervisor"

# "Summarizer": "grades the information provided by the members and generates a summary in clear language before reply",
members = {
    "Researcher": "searchs information about the user request related to the event COP16 and generates a response",
    "VacationsPlanner": "helps people find their vacations and buy vacations plans",
    worker_id: "helps people find what to do in cities"
}
agent_supervisor = AgentSupervisor(model=model, members=members)
research_graph = StateGraph(AgentState)
# research_graph.add_node("CustomerAgent", supervisor_nodes.reply_to_user)
research_graph.add_node("Researcher", process_request_crag_as_team)
research_graph.add_node("Assistant", supervisor_nodes.assistant)
research_graph.add_node("supervisor", agent_supervisor.supervisor_agent)
research_graph.add_node("reply", supervisor_nodes.gen_final_reply)
research_graph.add_node("VacationsPlanner", process_request_vacations_planner_as_team)
research_graph.add_node(worker_id, dynamic_worker.process_request_as_agent)

# Define the control flow
research_graph.add_edge("Researcher", "supervisor")
research_graph.add_edge("VacationsPlanner", "supervisor")
research_graph.add_edge(worker_id, "supervisor")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"Researcher": "Researcher", "VacationsPlanner": "VacationsPlanner", worker_id: worker_id, "FINISH": "reply"},
)
# research_graph.add_edge(START, "supervisor")

research_graph.add_edge(START, "Assistant")
research_graph.add_conditional_edges(
    "Assistant",
    decide_to_reply,
    {
        "supervisor": "supervisor",
        "FINISH": "reply",
    },
)

research_graph.add_edge("reply", END)



# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


def init_conversation(message: str):
    with RedisSaver.from_conn_info(host="localhost", port=6379, db=1) as checkpointer:
        chain = research_graph.compile(
            checkpointer=checkpointer
        )

        research_chain = enter_chain | chain
        reply = research_chain.invoke(
            message,
            {
                "recursion_limit": 150,
                "user_id": "restebance@gmail.com",
                "thread_id": "11"
            },
        )
        print(reply)
        # print("stream: ", s.keys())
        # if 'reply' in s.keys():
        #     print("reply found")
        #     print("DONE: ", s["reply"]["cool_post"])
        # elif 'CustomerAgent' in s.keys():
        #     print(s['CustomerAgent'])
        # else:
        #     print("END")

if __name__ == "__main__":
    # init_conversation("Hola")
    # init_conversation("que planes tienes disponibles?")
    # init_conversation("puedes darme los planes mas bonitos?")
    # init_conversation("Mi nombre es Esteban, tengo 33 años y mi id es 123")
    # init_conversation("que planes tienes disponibles para la ciudad?")
    init_conversation("amplia información sobre el plan TorreAlta")