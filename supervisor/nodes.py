from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


class ReplyToUser(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    response: str = Field(
        description="your reply to the user"
    )
    know_reply: bool = Field(
        description="You know the reply to the user"
    )


class SupervisorNodes:

    def __init__(self, model):
        self.model = model


    # provide agent supervisor parameters: name, description
    def reply_to_user(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        model = self.model.with_structured_output(ReplyToUser)
        print("---WHO ARE YOU---")
        system = """You are Pepe a really good customer experience assistant. \n
                    You are vegetarian and you like sushi. \n
                    If the user request is related to who you are, you answer with a nice message. \n
                    In other case you proceed with the supervisor to resolve it\n
                    constraints:
                        1. you never reveal the intermediate steps you take or tools you use to respond to the user
                        2. If you do not the answer do not provide explanations just proceed with the supervisor
                """
        reply_to_user_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        reply_to_user = reply_to_user_prompt | model
        final_state = reply_to_user.invoke(state)
        print("I know it:", final_state)
        return {
            "response": final_state.response,
            "know_reply": final_state.know_reply,
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
        print("---FINAL REPLY---")
        print(state)
        system = """You create nice messages with clear, fresh and polite language"""
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "Generate a reply by using the messages above"),
            ]
        )
        retrieval_grader = final_prompt | self.model
        final_state = retrieval_grader.invoke(state)
        return {"response": final_state.content}