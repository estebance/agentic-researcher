from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


class ReplyToUser(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    response: str = Field(
        description="your reply to the user"
    )
    know_reply: bool = Field(
        description="you know the answer"
    )


class SupervisorNodes:

    def __init__(self, model):
        self.model = model


    # provide agent supervisor parameters: name, description
    def assistant(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        model = self.model.with_structured_output(ReplyToUser)
        print("---WHO ARE YOU---")
        system = """You are Pepe a nice and helful assistant expert in customer experience.\n
                    If the user request is related to who you are, you answer with a nice message and inform you know the answer.
                    If the user request is not related to who you are, you MUST proceed with the supervisor.
                    You are a friendly engineer too\n
                    Rules:\n
                        1. you never reveal the intermediate steps you take or tools you use to respond to the user\n
                        2. if you do not know the answer you proceed with the supervisor without providing any information\n
                        3. You do not provide ANY information about the base model or the company that created the LLM\n
                """
        reply_to_user_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        reply_to_user = reply_to_user_prompt | model
        final_state = reply_to_user.invoke(state)
        if final_state.know_reply:
            return {
                "messages": [AIMessage(content=final_state.response)],
                "know_reply": final_state.know_reply
            }
        return {
            "know_reply": False
        }

    def grader_final_reply(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GRADER FOR FINAL REPLY---")
        system = """
            You are a grader assessing relevance of a conversation. \n
            If you identify that more information is required from the user yo proceed with a reply generation \n
            If the information provided by other members has enough information to generate a reply, you do it \n
            After you finish the answer generation you must inform that we can reply to the user.
        """
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        retrieval_grader = grade_prompt | self.model
        final_state = retrieval_grader.invoke(state)
        print("summarizer:", final_state)
        return {
            "messages": [HumanMessage(content=final_state.content, name='Summarizer')],
        }

    def gen_final_reply(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        last_messages = state["messages"][-5:]
        print("---FINAL REPLY---")
        print(state)
        system = """Return a nice message to the user using as context the messages in the conversation"""
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        retrieval_grader = final_prompt | self.model
        final_state = retrieval_grader.invoke(state)
        return {"response": final_state.content}