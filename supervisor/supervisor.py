from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import Literal
import langchain

langchain.verbose = True
from langchain_anthropic import ChatAnthropic

# IMPORTANT: this thing is key because it defines who goes next


class AgentSupervisor:

    # The Researcher: searchs information about the user request related to the event COP16 and generates a response
    # The Summarizer: grades the information provided by the researcher and generates a summary in clear language
    def __init__(self, model, members: dict):
        self.model = model
        self.members = members
        members_names = []
        members_descriptions = ""
        for key in members.keys():
            members_names.append(key)
            members_descriptions = members_descriptions + "\n" +  f"{key}:{members[key]}"
        self.system_prompt = """
            You are a supervisor tasked with managing a conversation between the
            following workers:  {members}.
            Each one of these workers has the following roles:
            {members_descriptions}
            Given the following user request, respond with the worker to act next if is necessary or to finish and respond to the user. Each worker will perform a
            task and respond with their results.
            Call the same worker more than twice is not allowed at some point you must respond with FINISH.
        """ %()
        self.members_options = ["FINISH"] + members_names
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "human",
                    "Given this conversation, who should act next or should we FINISH? Select one of: {options}",
                ),
            ]
        ).partial(options=str(self.members_options), members=", ".join(members_names), members_descriptions=members_descriptions)

    def supervisor_agent(self, state):
        member_options = self.members_options
        # check how to do it better
        class routeResponse(BaseModel):
            next: Literal[*member_options]
        supervisor_chain = self.prompt | self.model.with_structured_output(routeResponse)
        supervision = supervisor_chain.invoke(state)
        return supervision

