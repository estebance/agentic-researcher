### Question Re-writer
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QuestionRewriter:

    def __init__(self):
        self.model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
        system = """
            You a question re-writer that converts an input question to a better version that is optimized \n
            for web search. Look at the input and try to reason about the underlying semantic intent / meaning.\n
            only return the converted question no additional details must be included
        """
        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
            ]
        )


    def gen_question_rewriter_chain(self):
        question_rewriter = self.rewrite_prompt | self.model | StrOutputParser()
        return question_rewriter
        # question_rewriter.invoke({"question": question})


