import os
from langchain.schema import Document
from .generator import Generator
from .graph_state import GraphState
from .question_rewriter import QuestionRewriter
from .retrieval_grader import RetrievalGrader
from .tools import retrieve_web_search_tool, retrieve_bedrock_kdb
from .nodes import CragNodes
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph, START
from IPython.display import Image


class WorkflowGraph:

    def __init__(self, model, kdb_retriever_params, web_retriever_params):
        self.nodes = CragNodes(model, kdb_retriever_params, web_retriever_params)
        self.model = model
        # COMPILE THE GRAPH
        workflow = StateGraph(GraphState)
        # Define the nodes
        workflow.add_node("rewrite", self.nodes.rewrite)
        workflow.add_node("retrieve", self.nodes.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.nodes.grade_documents)  # grade documents
        workflow.add_node("generate", self.nodes.generate)  # generatae
        workflow.add_node("web_search_node", self.nodes.web_search)  # web search
        # Build graph
        workflow.add_edge(START, "rewrite")
        workflow.add_edge("rewrite", "retrieve")
        # workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.nodes.decide_to_generate,
            {
                "web_search_node": "web_search_node",
                "generate": "generate",
            },
        )
        # workflow.add_edge("transform_query", "web_search_node")
        workflow.add_edge("web_search_node", "generate")
        workflow.add_edge("generate", END)
        # Compile
        self.workflow = workflow


    def generate_graph(self):
        try:
            image = Image(self.workflow.get_graph(xray=True).draw_mermaid_png())
            with open("crag_image1.png", "wb") as fout:
                fout.write(image.data)
            # display(image)
        except Exception:
            # This requires some extra dependencies and is optional
            pass

    # TODO check future usage
    # def invoke_crag(self, app, question):
    #     final_response = ""
    #     inputs = {"question": question}
    #     for output in app.stream(inputs):
    #         for key, value in output.items():
    #             # Node
    #             print(f"Node '{key}':")
    #             # Optional: print full state at each node
    #             # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    #         print("\n---\n")
    #
    #         # Final generation
    #         print(output.keys())
    #         if "generate" in output.keys():
    #             final_response = output["generate"]["generation"]
    #     print(final_response)
    #     return final_response