from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
import logging
import json
from pydantic import BaseModel
from dotenv import load_dotenv
from state import State
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic

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

@tool
def reformulate_question(location: str, state: State):
    """Call to reformulate the question based on the chat history before try to generate a response."""
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
    rewritten_question_message = (f"Given a chat history and the latest user question \n\
        which might reference context in the chat history, formulate a standalone question\n\
        which can be understood without the chat history. Do NOT answer the question,\n\
        just reformulate it if needed and otherwise return it as is."
    )
    messages = state["messages"] + [HumanMessage(content=rewritten_question_message)]
    response = model.invoke(messages)
    print(f"tool response: {response.content}")
    return "content"
