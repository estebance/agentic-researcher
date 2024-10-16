from dotenv import load_dotenv
load_dotenv()

from crag.crag import invoke_crag

my_question = "Que eventos hay en la COP 16?"

invoke_crag(my_question)