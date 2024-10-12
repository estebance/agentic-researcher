from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
import logging
import json
from pydantic import BaseModel
from dotenv import load_dotenv
from state import State
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

load_dotenv()

class AskHuman(BaseModel):
    """A tool to let the user approve the usage of any other tools"""
    approval: str


web_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True
)