from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
import logging
import json
from dotenv import load_dotenv

load_dotenv()


web_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_domains=[
        "https://www.cop16colombia.com/es/agenda-oficial-cop16/agenda-zona-verde/evento-19-20-oct"
        "https://www.cop16colombia.com/es/agenda-oficial-cop16/agenda-zona-verde/eventos-21-octubre"
    ]
)