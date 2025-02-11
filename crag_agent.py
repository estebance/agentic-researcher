from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrock
from crag.workflow import WorkflowGraph
from services.kdbs.kdb_retriever import KDBRetriever
from services.llms.model_selector import ModelSelector
from services.redis_checkpointer.redis_saver import RedisSaver
from config_retriever import ConfigRetriever
# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.

class CragAgent:

    def __init__(self):
        parameters_file_name  = "params.json"
        config_retriever = ConfigRetriever(parameters_file_name)
        kdb_retriever = config_retriever.retriever
        model = config_retriever.model
        self.checkpointer = config_retriever.checkpointer
        self.graph = WorkflowGraph(model, kdb_retriever, None)

    def print_workflow(self):
        self.graph.generate_graph()

    def process_request_crag(self, user_id, thread_id, human_message):
        pass
        # LOAD CONFIG THEN BUILD WORKFLOW AND INVOKE
        # workflow = graph.workflow
        # graph.generate_graph()
        with RedisSaver.from_conn_info(self.checkpointer) as checkpointer:
                llm_app = self.graph.workflow.compile(
                    checkpointer=checkpointer,
                )
                config = {
                    "configurable": {
                        # The passenger_id is used in our flight tools to
                        # fetch the user's flight information
                        "user_id": user_id,
                        # Checkpoints ar2e acqcessed by thread_id
                        "thread_id": thread_id,
                    }
                }
                message_inputs = [HumanMessage(content=human_message)]
                final_state = llm_app.invoke(
                    {"messages": message_inputs}, config
                )
                print("final response")
                print(final_state['generation'])
                return final_state["generation"]


# def process_request_crag_as_team(agent_name, state):
#     config_parameters = retrieve_parameters()
#     print(config_parameters)
#     model = ChatAnthropic(model=config_parameters.llm_model_id, temperature=0)
#     # model = ChatBedrock(model_id=config_parameters.llm_model_id, temperature=0)
#     # LOAD CONFIG THEN BUILD WORKFLOW AND INVOKE
#     graph = WorkflowGraph(model, config_parameters.researcher_worker.kdb_retriever_params, config_parameters.researcher_worker.web_retriever)
#     workflow = graph.workflow
#     # OH MY CAT there is not checkpointer here
#     llm_app = workflow.compile()
#     final_state = llm_app.invoke(state)
#     # print("final response: ", final_state)
#     print("final_message: ", final_state["generation"])
#     # returns as human
#     return {
#         "messages": [HumanMessage(content=final_state["generation"], name=agent_name)]
#     }

if __name__ == "__main__":
    agent = CragAgent()
    agent.print_workflow()
    user_id = "restebance@gmail.com"
    thread_id = "15"
    human_meesage = "Hi there"
    agent.process_request_crag(user_id, thread_id, human_meesage)