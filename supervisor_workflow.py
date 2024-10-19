import functools
import operator
from typing import Sequence
from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_core.messages import BaseMessage
from supervisor.supervisor import AgentSupervisor
from config import retrieve_parameters
from langchain_aws import ChatBedrock
from crag_agent import process_request_crag_as_team
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage
from services.redis_checkpointer.redis_saver import RedisSaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Team
    team_members: List[str]
    # The 'next' field indicates where to route to next
    next: str


config_parameters = retrieve_parameters()
model = ChatBedrock(model_id=config_parameters.model_id, temperature=0)


# test node
def generate_final_reply(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GRADER FOR FINAL REPLY---")
    messages = state["messages"]
    system = """You are a grader assessing relevance of a conversation. \n
                If the information provided by the Researcher has enough information to generate a summary you do it \n
                If the participants mention that they do not know a response you generate the summary\n
                After you finish the summary you say that we can reply to the user.
                If you see that a participant could provide more information you do not mention thatwe can reeply to the user.
    """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Do your job"),
        ]
    )
    retrieval_grader = grade_prompt | model
    final_state = retrieval_grader.invoke(state)
    print("summarizer:", final_state)
    return {
        "messages": [HumanMessage(content=final_state.content, name='Summarizer')]
    }


agent_supervisor = AgentSupervisor(model=model, members=['Researcher', 'Summarizer'])


research_graph = StateGraph(AgentState)
research_graph.add_node("Researcher", process_request_crag_as_team)
research_graph.add_node("Summarizer", generate_final_reply)
research_graph.add_node("supervisor", agent_supervisor.supervisor_agent)

# Define the control flow
research_graph.add_edge("Researcher", "supervisor")
research_graph.add_edge("Summarizer", "supervisor")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"Researcher": "Researcher", "Summarizer": "Summarizer", "FINISH": END},
)
research_graph.add_edge(START, "supervisor")
with RedisSaver.from_conn_info(host="localhost", port=6379, db=1) as checkpointer:
    chain = research_graph.compile(
        checkpointer=checkpointer
    )


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


research_chain = enter_chain | chain
for s in research_chain.stream(
    "Que eventos tenemos para la COP16?",
    {
        "recursion_limit": 150,
        "user_id": "393939393212323232323233234233434345345334334",
        "thread_id": "3949439349123222344we34433333123834534534445"
    },
):
    print("stream: ", s)
    if "__end__" not in s:
        print(s)
        print("---")
    else:
        print("END")