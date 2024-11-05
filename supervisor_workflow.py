from functools import partial
from typing import Sequence, Literal
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
from langchain_core.messages import HumanMessage, RemoveMessage
from services.redis_checkpointer.redis_saver import RedisSaver
from vertex import load_vertex_model_gemini
import json
from pydantic import BaseModel, Field
from IPython.display import Image
from dotenv import load_dotenv

load_dotenv()


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
    explanation: str
    cool_post: str

def filter_conversation(state: AgentState):
    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last ten messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-10]]
    return {"messages": delete_messages}


def should_filter_conversation(state: AgentState) -> Literal["filter_conversation", "supervisor"]:
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 15:
        return "filter_conversation"
    else:
        return "supervisor"

# class Workflow
class SupervisorWorkflow:


    def __init__(self):
        self.config_parameters = retrieve_parameters()
        # retrieve nodes
        self.worker_model = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0
        )
        self.supervisor_nodes = SupervisorNodes(self.worker_model)
        self.workflow_members, self.action_map = self.extract_workflow_members_confg()
        # inject members
        self.workflow_members.update({
            self.config_parameters.assistant_worker.id: self.config_parameters.assistant_worker.task
        })
        self.enabled_researcher = self.config_parameters.researcher_worker.enabled
        if self.enabled_researcher:
            self.workflow_members.update({self.config_parameters.researcher_worker.id: self.config_parameters.researcher_worker.task})
        self.agent_supervisor = AgentSupervisor(model=self.worker_model, members=self.workflow_members)
        self.graph = StateGraph(AgentState)


    def extract_workflow_members_confg(self):
        dynamic_members = {}
        dynamic_action_map = {}
        for worker in self.config_parameters.workers:
            dynamic_members[worker.id] = worker.task
            dynamic_action_map[worker.id] = worker.id
        return dynamic_members, dynamic_action_map


    def inject_nodes(self, graph, path_map_param):
        for worker in self.config_parameters.workers:
            worker_tools = []
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
            dynamic_worker = DynamicWorker(worker.id, worker.name, worker.task, self.worker_model, tools=worker_tools)
            graph.add_node(worker.id, dynamic_worker.process_request_as_agent)
            graph.add_edge(worker.id, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            path_map_param,
        )
        return graph

    def gen_workflow(self):

        self.graph.add_node("supervisor", self.agent_supervisor.supervisor_agent)
        self.graph.add_node("filter_conversation", filter_conversation)
        # assistant always enters
        self.graph.add_node("AssistantWorker", self.supervisor_nodes.assistant)

        self.graph.add_node("reply", self.supervisor_nodes.gen_final_reply)
        self.action_map.update({"FINISH": "reply"})
        # ALWAYS GOES
        self.graph.add_edge("AssistantWorker", "supervisor")
        # self.workflow_members.update({"AssistantWorker": "introduces the agent and provides details about the agent (name, role and features)"})
        self.action_map.update({"AssistantWorker": "AssistantWorker"})

        if self.enabled_researcher:
            process_request_crag_as_team_partial = partial(process_request_crag_as_team, self.config_parameters.researcher_worker.id)
            self.graph.add_node(self.config_parameters.researcher_worker.id, process_request_crag_as_team_partial)
            self.graph.add_edge(self.config_parameters.researcher_worker.id, "supervisor")
            self.action_map.update({self.config_parameters.researcher_worker.id:self.config_parameters.researcher_worker.id})
        self.graph = self.inject_nodes(self.graph, self.action_map)
        self.graph.add_conditional_edges(
            START,
            should_filter_conversation,
            {"supervisor": "supervisor", "filter_conversation": "filter_conversation"}
        )
        self.graph.add_edge("filter_conversation", "supervisor")
        self.graph.add_edge("reply", END)


    def gen_workflow_image(self):
        try:
            compiled_workflow= self.graph.compile()
            image = Image(compiled_workflow.get_graph(xray=True).draw_mermaid_png())
            with open("supervisor.png", "wb") as fout:
                fout.write(image.data)
            # display(image)
        except Exception:
            # This requires some extra dependencies and is optional
            pass

    def gen_chain(self):
        self.gen_workflow()
        with RedisSaver.from_conn_info(host="localhost", port=6379, db=1) as checkpointer:
            supervised_chain = self.graph.compile(
                checkpointer=checkpointer
            )
            supervised_chain = enter_chain | supervised_chain
            return supervised_chain


def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


def gen_chain(graph):
    with RedisSaver.from_conn_info(host="localhost", port=6379, db=1) as checkpointer:
        supervised_chain = graph.compile(
            checkpointer=checkpointer
        )
        supervised_chain = enter_chain | supervised_chain
        return supervised_chain


def init_conversation(graph , message: str):
    with RedisSaver.from_conn_info(host="localhost", port=6379, db=1) as checkpointer:
        chain = graph.compile(
            checkpointer=checkpointer
        )
        research_chain = enter_chain | chain

        # print("stream: ", s.keys())
        # if 'reply' in s.keys():
        #     print("reply found")
        #     print("DONE: ", s["reply"]["cool_post"])
        # elif 'CustomerAgent' in s.keys():
        #     print(s['CustomerAgent'])
        # else:
        #     print("END")

supervised_workflow = SupervisorWorkflow()
supervised_workflow.gen_workflow()
# supervised_workflow.gen_workflow_image()

if __name__ == "__main__":
    graph = supervised_workflow.graph
    chain = gen_chain(graph)
    # message = "me gustaria saber planes para la ciudad de Medellin?"
    message = " que planes tenes en medellin??"
    reply = chain.invoke(
        message,
        {
            "recursion_limit": 150,
            "user_id": "restebance@gmail.com",
            "thread_id": "15"
        },
    )
    print(reply['response'])
    # init_conversation("Que sabes hacer?")
    # init_conversation("Quien eres?")
    # init_conversation("que planes vacacionales tienes disponibles?")
    # init_conversation("puedes darme los planes mas bonitos?")
    # init_conversation("Mi nombre es Esteban, tengo 33 años y mi id es 123")
    # init_conversation("que planes tienes disponibles para la ciudad?")
    #   init_conversation("solo dame la información")
    # init_conversation("amplia información sobre el plan TorreAlta") el plan TorreAlta")