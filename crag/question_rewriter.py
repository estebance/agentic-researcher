### Question Re-writer
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

    def gen_rewriter_chain(self):
        structured_llm = self.model.with_structured_output(GeneratorQuestion)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        Given a chat history and the latest user question \n\
                        which might reference context in the chat history, formulate a standalone question\n\
                        which can be understood without the chat history. Do NOT answer the question,\n\
                        just reformulate it if needed and otherwise return it as is and proceed. \n\
                        Do NOT provide explanations just return the question""",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        rewriter_node =  prompt | structured_llm
        return rewriter_node


