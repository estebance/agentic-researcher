import operator
from datetime import datetime
import pytz
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import BaseMessage, AIMessage
from typing import Sequence
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    response: str
    know_reply: bool


class AgentIntroduction(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    response: str = Field(
        description="your reply to the user"
    )
    know_reply: bool = Field(
        description="You know the reply to the user"
    )

class AgentSupervisor:

    def __init__(self, model, name, role, features, language):
        self.model = model
        self.name = name
        self.role = role
        self.features = features
        self.language = language


    def agent_node(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state
            name: (str): agent name
            role: (str): agent role
            features: (str) agent features
            language: (str) agent language
        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        current_time = datetime.now(pytz.timezone('America/Bogota')).strftime('%Y-%m-%d %H:%M:%S')
        structured_model = self.model.with_structured_output(AgentIntroduction)
        print("---AGENT---")
        system = f"""
            You are {self.name} a really good {self.role}. \n
            These are your main characteristics: \n
            {self.features}\n
            The current time is: {current_time} \n
            Your language is: {self.language} \n
            When there is an user request and you do not know the answer yo proceed with the supervisor to resolve it\n
            Use only information provided from the context to generate a response\n
        """
        agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        reply_to_user_chain = agent_prompt | structured_model
        model_response = reply_to_user_chain.invoke(state)
        print(model_response)
        return {
            "response": model_response.response,
            "know_reply": model_response.know_reply
        }