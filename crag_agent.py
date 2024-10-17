from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from crag.crag import workflow
from services.redis_checkpointer.redis_saver import RedisSaver
# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
def process_request_crag(user_id, thread_id, human_message):
    with RedisSaver.from_conn_info(host="localhost", port=6379, db=1) as checkpointer:
        llm_graph = workflow.compile(
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
        final_state = llm_graph.invoke(
            {"messages": message_inputs}, config
        )
        print("final response")
        print(final_state['generation'])
        return final_state["generation"]