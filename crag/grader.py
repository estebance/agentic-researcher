from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class Grader:

    def __init__(self):
        self.model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
        self.structured_llm_grader = self.model.with_structured_output(GradeDocuments)
        system = """You are a grader assessing relevance of a retrieved document to an user question. \n
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )


    def gen_retrieval_grader(self):
        retrieval_grader = self.grade_prompt | self.structured_llm_grader
        return retrieval_grader