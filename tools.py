from langchain_core.tools import tool
import logging
import json


@tool
def retrieve_with_vectorstore(user_email: str, request_id: str):
    """

    """

@tool
def retrieve_with_web_search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."