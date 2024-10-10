from dotenv import load_dotenv
load_dotenv()

from crag.crag import invoke_crag

my_question = "Que eventos tenemos programados para la COP16?"

invoke_crag(my_question)