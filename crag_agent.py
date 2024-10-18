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
PARAMETERS_FILE = "params.json"


def process_request_crag(user_id, thread_id, human_message):

    config_parameters = retrieve_parameters()
    print(config_parameters)
    # model = ChatAnthropic(model=config_parameters.model_id, temperature=0)
    model = ChatBedrock(model_id=config_parameters.model_id, temperature=0)
    # LOAD CONFIG THEN BUILD WORKFLOW AND INVOKE
    graph = WorkflowGraph(model, config_parameters.knowledge_base_id)
    workflow = graph.workflow
    with RedisSaver.from_conn_info(host="localhost", port=6379, db=1) as checkpointer:
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


def process_request_crag_as_team(message):
    config_parameters = retrieve_parameters()
    print(config_parameters)
    # model = ChatAnthropic(model=config_parameters.model_id, temperature=0)
    model = ChatBedrock(model_id=config_parameters.model_id, temperature=0)
    # LOAD CONFIG THEN BUILD WORKFLOW AND INVOKE
    graph = WorkflowGraph(model, config_parameters.knowledge_base_id)
    workflow = graph.workflow
    # OH MY CAT there is not checkpointer here
    llm_app = workflow.compile()
    # each interaction enters as HUMAN
    message_inputs = [HumanMessage(content=message)]
    final_state = llm_app.invoke(
        {"messages": message_inputs}
    )
    return final_state