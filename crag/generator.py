### Generate
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

class Generator:

    def __init__(self, model):
        self.model = model
        system_prompt = """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\n
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n
            You always use the provided language {language}\n
            <context>
                {context}
            </context>
        """
        human_prompt = "Now, answer this question using the above context: {question}"
        self.generator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )

    def gen_rag_chain(self):
        question_answer_chain = create_stuff_documents_chain(self.model, self.generator_prompt)
        return question_answer_chain