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



model = ChatBedrock(model_id="claude-3-5-sonnet-20241022", temperature=0)
members = {
    "Researcher": "searchs information about the user request related to the event COP16 and generates a response",
    "Summarizer": "grades the information provided by the researcher and generates a summary in clear language",
    "VacationsPlanner": "You like to plan vacations"
}
agent_supervisor = AgentSupervisor(model, members)
