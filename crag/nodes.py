# based on chat history write a new question
### Question Re-writer
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chatanthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

# Data model
class GeneratorQuestion(BaseModel):
    """generated question"""
    question: str = Field(
        description="generated question",
    )

def gen_rewriter_chain():
    structured_llm = model.with_structured_output(GeneratorQuestion)
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

def rewrite(state):
    """
    Rewrite documents
    Args:
        state (dict): The current graph state
    Returns:
        question (str): Reformulated question
    """
    rewriter_chain = gen_rewriter_chain()
    print("---REWRITE---")
    messages = state["messages"]
    response = rewriter_chain.invoke({"messages": messages})
    print(response.question)
    return {"question": response.question}
