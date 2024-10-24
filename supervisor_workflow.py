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
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent

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
    messages: Annotated[Sequence[BaseMessage], operator.add]
    #
    response: str
    know_reply: bool
    # Team
    team_members: List[str]
    # The 'next' field indicates where to route to next
    next: str
    summary: str
    cool_post: str


config_parameters = retrieve_parameters()

# from parameters we are going to retrieve
# custom agent name and custom agent role description
# pass the parameters to the simple_tool workflow as an agent to resolve a request
# The parameters are (tool name, tool_description, tool_schema)




model = ChatBedrock(model_id=config_parameters.llm_model_id, temperature=0)

# role descriptor
def reply_to_user(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    model = ChatBedrock(model_id=config_parameters.llm_model_id, temperature=0).with_structured_output(ReplyToUser)
    print("---WHO ARE YOU---")
    system = """You are Pepe a really good customer experience assistant. \n
                You are vegetarian and you like sushi. \n
                When there is an user request yo do not know the answer yo proceed with the supervisor to resolve it\n
                Use only information provided from the context to generate a response\n"""
    reply_to_user_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    reply_to_user = reply_to_user_prompt | model
    final_state = reply_to_user.invoke(state)
    print("I know it:", final_state)
    return {
        "response": final_state.response,
        "know_reply": final_state.know_reply,
    }


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
                If you see that a participant could provide more information you do not mention that we can reply to the user.
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
        "messages": [HumanMessage(content=final_state.content, name='Summarizer')],
        "summary": final_state.content
    }

def decide_to_reply(state):
    if state["response"] and state["know_reply"]:
        return "FINISH"
    else:
        return "research"


def respond_to_the_user(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---FINAL REPLY---")
    summary = state["summary"]
    print("for final reply: ", summary)
    system = """You are a journalist who creates nice posts with clear, fresh and polite language"""
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Generate a nice post by using the following summary: {summary}"),
        ]
    )
    retrieval_grader = final_prompt | model
    final_state = retrieval_grader.invoke({"summary": summary})
    print("cool post: ", final_state.content)
    return {"cool_post": final_state.content}

agent_supervisor = AgentSupervisor(model=model, members=['Researcher', 'Summarizer'])
research_graph = StateGraph(AgentState)
research_graph.add_node("CustomerAgent", reply_to_user)
research_graph.add_node("Researcher", process_request_crag_as_team)
research_graph.add_node("Summarizer", generate_final_reply)
research_graph.add_node("supervisor", agent_supervisor.supervisor_agent)
research_graph.add_node("reply", respond_to_the_user)


# Define the control flow
research_graph.add_edge("Researcher", "supervisor")
research_graph.add_edge("Summarizer", "supervisor")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"Researcher": "Researcher", "Summarizer": "Summarizer", "FINISH": "reply"},
)
# research_graph.add_edge(START, "supervisor")

research_graph.add_edge(START, "CustomerAgent")
research_graph.add_conditional_edges(
    "CustomerAgent",
    decide_to_reply,
    {
        "research": "supervisor",
        "FINISH": END,
    },
)

research_graph.add_edge("reply", END)
# with RedisSaver.from_conn_info(host="localhost", port=6379, db=1) as checkpointer:
#     chain = research_graph.compile(
#         checkpointer=checkpointer
#     )


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
# def enter_chain(message: str):
#     results = {
#         "messages": [HumanMessage(content=message)],
#     }
#     return results
#
#
# research_chain = enter_chain | chain
# for s in research_chain.stream(
#     "Cuentame sobre ti?",
#     {
#         "recursion_limit": 150,
#         "user_id": "2344",
#         "thread_id": "2Q3423223"
#     },
# ):
#     print("stream: ", s.keys())
#     if 'reply' in s.keys():
#         print("reply found")
#         print("DONE: ", s["reply"]["cool_post"])
#     else:
#         print("END")