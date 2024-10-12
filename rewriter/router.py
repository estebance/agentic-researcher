### Router

from typing import Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# Data model
class RouteQuery(BaseModel):
    """Route an user question if it is not clear"""

    datasource: Literal["rewriter", "assistant"] = Field(
        ...,
        description="Given a user question choose to route it to the rewriter or proceed.",
    )


# LLM with function call
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
structured_llm_router = model.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to be rewritten or to proceed with its resolution.
The rewriter evaluates the question and by using the chat history try to improve the question.
Use the assistant to proceed"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router