# based on chat history write a new question
### Question Re-writer
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain.schema import Document
from pydantic import BaseModel, Field
from .question_rewriter import QuestionRewriter
from .retrieval_grader import RetrievalGrader
from .generator import Generator
from .tools import retrieve_web_search_tool, retrieve_bedrock_kdb

# KDB
BEDROCK_KDB = os.environ.get('BEDROCK_KDB')
# Data model
class GeneratorQuestion(BaseModel):
    """generated question"""
    question: str = Field(
        description="generated question",
    )

class CragNodes:

    def __init__(self, model, kdb_retriever_params, web_retriever_params):
        self.model = model
        self.kdb_retriever_params = kdb_retriever_params
        self.web_retriever_params = web_retriever_params


    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]
        # Retrieval
        bedrock_kdb_retriever = retrieve_bedrock_kdb(self.kdb_retriever_params)
        documents = bedrock_kdb_retriever.invoke(question)
        return {"documents": documents, "question": question}


    def rewrite(self, state):
        """
        Rewrite documents
        Args:
            state (dict): The current graph state
        Returns:
            question (str): Reformulated question
        """
        question_rewriter = QuestionRewriter(self.model)
        rewriter_chain = question_rewriter.gen_rewriter_chain()
        print("---REWRITE---")
        messages = state["messages"]
        response = rewriter_chain.invoke({"messages": messages})
        print(response.question)
        return {"question": response.question}


    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        language = "Spanish"
        # TODO CHECK THE IDEAL MANAGEMENT FROM THE STATE PERSPECTIVE
        # RAG generation
        generator = Generator(self.model)
        rag_chain = generator.gen_rag_chain()
        generation = rag_chain.invoke({"context": documents, "question": question, "language": language})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        retrieval_grader = RetrievalGrader(self.model)
        retrieval_grader_chain = retrieval_grader.gen_retrieval_grader_chain()
        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader_chain.invoke(
                {"question": question, "document": d.page_content}
            )
            # at least one document was not relevant
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}


    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        # Re-write question
        question_rewriter = QuestionRewriter(self.model)
        question_rewriter_chain = question_rewriter.gen_rewriter_chain()
        better_question = question_rewriter_chain.invoke({"question": question})
        return {"documents": documents, "question": better_question}


    def web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        print(question)
        # Web search
        web_search_tool = retrieve_web_search_tool(self.web_retriever_params)
        docs = web_search_tool.invoke({"query": question})
        print(docs)
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        return {"documents": documents}


    ### Edges
    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        print(state["question"])
        web_search = state["web_search"]
        print(state["documents"])
        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, WEB SEARCH---"
            )
            return "web_search_node"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"