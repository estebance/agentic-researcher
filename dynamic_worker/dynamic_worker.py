from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from .state import State
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# stdout_callback_handler = StdOutCallbackHandler()
# llm_model = ChatAnthropic(
#     model="claude-3-5-sonnet-20241022",
#     callbacks=[stdout_callback_handler]
# )

class DynamicWorker:
    # TODO polish this prompt
    def __init__(self, worker_id, worker_name, worker_task, llm_model, tools):
        system_prompt = f"""
            You are a {worker_name}. You {worker_task}.
            If you require more information from the user you explicitly inform that to the Supervisor,
            If you determine that you have an answer yo explictly inform that you have an answer
            Follow this format to notify the Supervisor with one of the following actions:
                - FINISH: supervisor should finish
                - EVALUATE: supervisor should check if other worker can continue with the task
                - FINISH_ASK_USER: supervidor should Finish because the user must provide addional information
            <notification supervisor_recommendation=action explanation=explain_your_decision/>
            Rules:
             1. Use ONLY the information provided by the tools to generate an answer
             2. Do not try to create your own responses, if you do not know something just notify to the supervisor that you do not know
        """
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        self.llm_model = llm_model.bind_tools(tools)
        self.tools = tools
        self.worker_id = worker_id
        self.worker_graph = self._gen_graph()

    def _chatbot(self, state: State):
        chat_model = self.prompt | self.llm_model
        response = chat_model.invoke(state["messages"])
        print(response)
        return {
            "messages": [response]
        }

    def _gen_graph(self):
        graph_builder = StateGraph(State)
        # The first argument is the unique node name
        # The second argument is the function or object that will be called whenever
        # the node is used.
        graph_builder.add_node("chatbot", self._chatbot)
        graph_builder.add_node("tools", ToolNode(self.tools))
        # add first and end nodes (edges)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        graph_builder.add_edge("tools", "chatbot")
        graph = graph_builder.compile()
        return graph

    def process_request_as_agent(self, state):
        dynamic_worker_response = self.worker_graph.invoke(state)
        dynamic_worker_response_content = dynamic_worker_response['messages'][-1].content
        # returns as human
        return {
            "messages": [HumanMessage(content=dynamic_worker_response_content, name=self.worker_id)]
        }