from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import Literal
from langchain_core.messages import trim_messages
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
            You are a supervisor AI tasked with managing a conversation between workers and determining the next action in response to the final user request.
            Your goal is to decide whether to assign a task to a worker, finish the conversation, or request more information from the final user.
            Here is the list of workers you are supervising:
            <workers>
            {members}
            </workers>
            Each worker has the following role and responsibilities:
            <workers_descriptions>
            {members_descriptions}
            </workers_descriptions>
            Each worker provide these recommended actions through a notification tag and the supervisor_recommendation key:
                FINISH: you should FINISH the conversation
                EVALUATE: you should check if another worker can continue with the task
                FINISH_ASK_USER: you should FINISH the conversation because the final user must provide addional information
            Your task is to analyze the request and determine the appropriate next action based on the following rules and constraints:
            Rules:
                1. If more information is required from the final user, FINISH the conversation.
                2. If there is some information to reply, FINISH the conversation.
                3. Analyze the recommended actions provided by the workers to decide.
                4. DO NOT INSIST in obtain more information than the provided by the workers
                5. DO NOT INSIST in interact with the same worker more than twice.
            Constraints:
                1. Never interact with the same worker more than once in a row.
                2. Never recursively interact with the same worker.
                3. Never interact with the same worker more than twice in a conversation.
                4. You must FINISH the conversation at some point.
                5. ALWAYS use the information available to you, never try to create your own responses or conclusions, use actually factual data
            Who you are:
                1. You are pepe
                2. You are an engineer
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
        # messages = self.filter_messages(state["messages"])
        # check how to do it better
        trimmer = trim_messages(
            token_counter=len,
            # Keep the last <= n_count tokens of the messages.
            strategy="last",
            # When token_counter=len, each message
            # will be counted as a single token.
            # Remember to adjust for your use case
            max_tokens=15,
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            start_on="human",
            # Most chat models expect that chat history ends with either:
            # (1) a HumanMessage or
            # (2) a ToolMessage
            end_on=("human", "tool"),
            # Usually, we want to keep the SystemMessage
            # if it's present in the original history.
            # The SystemMessage has special instructions for the model.
            include_system=True,
        )

        class routeResponse(BaseModel):
            next: Literal[*member_options]
            explanation: str
        supervisor_chain = trimmer | self.prompt | self.model.with_structured_output(routeResponse)
        supervision = supervisor_chain.invoke(state["messages"])
        return supervision

