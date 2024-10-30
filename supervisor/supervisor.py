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
            members_descriptions = members_descriptions + "\n" +  f"{key}: {members[key]}"

        self.system_prompt = """
            You are a supervisor AI tasked with managing a conversation between workers and determining the next action in response to a user request. Your goal is to decide whether to assign a task to a worker, finish the conversation, or request more information from the user.
            Here is the list of workers you are supervising:
            <workers>
            {members}
            </workers>
            Each worker has the following role and responsibilities:
            <workers_descriptions>
            {members_descriptions}
            </workers_descriptions>
            Each worker can provide the following recommended actions through a notification tag:
                FINISH: supervisor should finish
                EVALUATE: supervisor should check if another worker can continue with the task
                ASK: supervidor should Finish because the user must provide addional information
            Your task is to analyze the request and determine the appropriate next action based on the following rules and constraints:
                Rules:
                    1. If more information is required from the user, finish the interaction.
                    2. If there is enough information to reply, finish the interaction.
                    3. If neither of the above applies, assign the next task to the most appropriate worker.
                Constraints:
                    1. Never call the same worker more than once in a row.
                    1. Never call the same worker more than twice.
                    2. You must finish the conversation at some point.
                    To make your decision, follow these steps:
                    1. Analyze the user request and the available worker roles.
                    2. Determine if there is enough information to respond or if more information is needed from the user.
                    3. If more information is needed or if a response can be provided, decide to finish the interaction.
                    4. If neither of the above applies, select the most appropriate worker for the next task.
                        5. Keep track of how many times each worker has been called to ensure you don't exceed the limit.
            You can also evaluate the workers recommendations to decide the next action
        """
        # <decision>
        #     Action: [Choose one: "Assign to Worker", "Finish - Need More Information", or "Finish - Ready to Respond"]
        #     [If "Assign to Worker" is chosen, include the following:]
        #     Worker: [Name of the selected worker]
        #     Reason: [Brief explanation of why this worker was chosen]
        #     [If "Finish - Need More Information" is chosen, include the following:]
        #     Reason: [Explanation of what additional information is needed from the user]
        #     [If "Finish - Ready to Respond" is chosen, include the following:]
        #     Reason: [Explanation of why the conversation can be concluded]
        # </decision
        # self.system_prompt = """
        #     You are a supervisor tasked with managing a conversation between the
        #     following workers:  {members}.
        #     Each one of these workers has the following roles:
        #     {members_descriptions}
        #     Given the following user request, determine which worker should act next or finish the conversation,
        #     if the user has to provide more information finish the conversation.
        #     Rules:
        #         If more information is required from the user, finish the interaction.
        #         If there is enough information to reply, finish the interaction.
        #     Constraints:
        #         You never call the same worker more than twice. At some point, you must finish the conversation.
        # """
        self.members_options = ["FINISH"] + members_names
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "human",
                    "Given the above conversation, who should act next or should we FINISH? Select one of: {options}",
                ),
            ]
        ).partial(options=str(self.members_options), members=", ".join(members_names), members_descriptions=members_descriptions)

    def filter_messages(self, messages: list):
        # This is very simple helper function which only ever uses the last message
        return messages[-10:]

    def supervisor_agent(self, state):
        member_options = self.members_options
        messages = self.filter_messages(state["messages"])
        # check how to do it better
        class routeResponse(BaseModel):
            next: Literal[*member_options]
        supervisor_chain = self.prompt | self.model.with_structured_output(routeResponse)
        supervision = supervisor_chain.invoke(messages)
        return supervision

