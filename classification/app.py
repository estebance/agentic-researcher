from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from .model import ClassificationModel


class ClassificationApp:

    def __init__(self):
        self.model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0).with_structured_output(
            ClassificationModel
        )
        system_prompt = ChatPromptTemplate.from_template("""
            Extract the desired information from the following passage.
            Only extract the properties mentioned in the 'Classification' function.
            Passage:
            {input}
        """)
        self.classification_chain = system_prompt | self.model
