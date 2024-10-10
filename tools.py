from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
import logging
import json
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class AskHuman(BaseModel):
    """Ask the human for approval before try to use any other tool"""
    approval: str


web_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True
)

