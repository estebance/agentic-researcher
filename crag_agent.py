from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from crag.workflow import WorkflowGraph
from services.redis_checkpointer.redis_saver import RedisSaver
from config import retrieve_parameters
# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.


def process_request_crag(user_id, thread_id, human_message):

    config_parameters = retrieve_parameters()
    print(config_parameters)
    # model = ChatAnthropic(model=config_parameters.model_id, temperature=0)
    model = ChatBedrock(
        model_id=config_parameters.llm_model_id,
        temperature=0
    )
    # LOAD CONFIG THEN BUILD WORKFLOW AND INVOKE
    graph = WorkflowGraph(model, config_parameters.kdb_retriever_params, config_parameters.web_retriever)
    workflow = graph.workflow
    with RedisSaver.from_conn_info(host=config_parameters.checkpointer.endpoint, port=config_parameters.checkpointer.port, db=config_parameters.checkpointer.db_number) as checkpointer:
        llm_app = workflow.compile(
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


def process_request_crag_as_team(state, agent_name='Researcher'):
    config_parameters = retrieve_parameters()
    print(config_parameters)
    # model = ChatAnthropic(model=config_parameters.model_id, temperature=0)
    model = ChatBedrock(model_id=config_parameters.model_id, temperature=0)
    # LOAD CONFIG THEN BUILD WORKFLOW AND INVOKE
    graph = WorkflowGraph(model, config_parameters.knowledge_base_id)
    workflow = graph.workflow
    # OH MY CAT there is not checkpointer here
    llm_app = workflow.compile()
    final_state = llm_app.invoke(state)
    # print("final response: ", final_state)
    print("final_message: ", final_state["generation"])
    # returns as human
    return {
        "messages": [HumanMessage(content=final_state["generation"], name=agent_name)]
    }