from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from .state import  State
from .tools import tools
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

stdout_callback_handler = StdOutCallbackHandler()

llm_model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    callbacks=[stdout_callback_handler]
)
llm_model = llm_model.bind_tools(tools)


def chatbot(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
                You are a vacations planner. You can help the user to find plans, buy plans.
                If you require more information from the user you explicitly inform that to  Supervisor,
                If you determine that you have an answer yo explictly inform that you have a response
                Follow this format to notify the Supervisor with one of the following actions:
                    FINISH: supervisor should finish
                    EVALUATE: supervisor should check if other worker can continue with the task
                    FINISH_ASK_USER: supervisor should Finish because the user must provide addional information
                <notify supervisor_recommendation=action explanation=explain_your_decision/>
                """
             ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chat_model =  prompt | llm_model
    response = chat_model.invoke(state["messages"])
    print(response)
    return {
        "messages": [response]
    }


graph_builder = StateGraph(State)
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

# add first and end nodes (edges)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge("chatbot", END)



def process_request_vacations_planner_as_team(state, agent_name='VacationsPlannerWorker'):
    # compiled graph
    graph = graph_builder.compile()
    # message_inputs = [HumanMessage(content="Que planes de turismo tienes disponibles, mi nombre es Esteban, tengo 33 años y my id es 1?")]
    # graph.invoke({"messages": message_inputs})
    vacations_planner_response = graph.invoke(state)
    vacations_planner_message = vacations_planner_response['messages'][-1].content
    print()
    # returns as human
    return {
        "messages": [HumanMessage(content=vacations_planner_message, name=agent_name)]
    }