from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import Literal
from langchain_anthropic import ChatAnthropic

# IMPORTANT: this thing is key because it defines who goes next


class AgentSupervidor:

    def __init__(self, model, members: list):
        self.model = model
        self.members = members
        self.system_prompt = """
            You are a supervisor tasked with managing a conversation between the
            following workers:  {members}. Given the following user request,
            respond with the worker to act next. Each worker will perform a
            task and respond with their results and status. When finished,
            respond with FINISH.
        """
        self.members_options = ["FINISH"] + self.members
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(options=str(self.members_options), members=", ".join(self.members))

    def supervisor_agent(self, state):
        member_options = self.members_options
        # check how to do it better
        class routeResponse(BaseModel):
            next: Literal[*member_options]
        supervisor_chain = self.prompt | self.model.with_structured_output(routeResponse)
        return supervisor_chain.invoke(state)
