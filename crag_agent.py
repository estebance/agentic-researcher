from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrock
from crag.workflow import WorkflowGraph
from services.kdbs.kdb_retriever import KDBRetriever
from services.llms.model_selector import ModelSelector
from services.redis_checkpointer.redis_saver import RedisSaver
from config import retrieve_parameters
# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.

class CragAgent:

    def __init__(self):
        self.config_parameters = retrieve_parameters()
        model_params = self.config_parameters.llm_model_provider
        self.model = ModelSelector(
            model_params.provider,
            model_params.llm_model_id,
            model_params.temperature,
            model_params.max_tokens,
            **model_params.provider_args
        )
        kdb_params = self.config_parameters.researcher_worker.kdb_retriever_params
        self.kdb_retriever = KDBRetriever(kdb_params)
        self.bedrock_retriever = self.kdb_retriever.retrieve_bedrock_kdb()
        self.graph = WorkflowGraph(self.model, self.kdb_retriever, None)

    def print_workflow(self):
        self.graph.generate_graph()

    def process_request_crag(self, user_id, thread_id, human_message):
        pass
        # LOAD CONFIG THEN BUILD WORKFLOW AND INVOKE
        # workflow = graph.workflow
        # graph.generate_graph()
        # with RedisSaver.from_conn_info(host=config_parameters.checkpointer.endpoint, port=config_parameters.checkpointer.port, db=config_parameters.checkpointer.db_number, auth_params=config_parameters.checkpointer.auth_params) as checkpointer:
        #     llm_app = workflow.compile(
        #         checkpointer=checkpointer,
        #     )
        #     config = {
        #         "configurable": {
        #             # The passenger_id is used in our flight tools to
        #             # fetch the user's flight information
        #             "user_id": user_id,
        #             # Checkpoints ar2e acqcessed by thread_id
        #             "thread_id": thread_id,
        #         }
        #     }
        #     message_inputs = [HumanMessage(content=human_message)]
        #     final_state = llm_app.invoke(
        #         {"messages": message_inputs}, config
        #     )
        #     print("final response")
        #     print(final_state['generation'])
        #     return final_state["generation"]


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