### Question Re-writer
import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser


class GeneratorQuestion(BaseModel):
    """generated question"""
    question: str = Field(
        description="generated question"
    )

class QuestionRewriter:

    def __init__(self, model):
        self.model = model
        self.system_prompt = """
            Given a chat history and the latest user question \n\
            which might reference context in the chat history, formulate a standalone question\n\
            which can be understood without the chat history. Do NOT answer the question,\n\
            just reformulate it if needed and otherwise return it as is and proceed. \n\
            Do NOT provide explanations just return the question
        """
        structured_llm = self.model.with_structured_output(GeneratorQuestion)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.system_prompt
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        rewriter_chain =  prompt | structured_llm
        self.rewriter_chain = rewriter_chain