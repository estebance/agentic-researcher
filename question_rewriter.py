
### Answer Grader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

# Data model
class RewriteQuestion(BaseModel):
    """question generated from chat history"""
    question: str = Field(
        description="improved question"
    )


# LLM with function call
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
structured_llm_grader = llm.with_structured_output(RewriteQuestion)

def invoke_rewriter_chain(messages):
    # Prompt
    # print(messages)
    all_messages = []
    for message in messages:
        if isinstance(message, AIMessage):
            all_messages.append(("ai", f"{message.content}"))
        elif isinstance(message, HumanMessage):
            all_messages.append(("human", f"{message.content}"))
    messages = ChatPromptTemplate.from_messages(
        all_messages +
        [("user", "{input}")]
    )
    rewriter_chain = messages | structured_llm_grader
    return rewriter_chain
