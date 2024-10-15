from classification.app import ClassificationApp
from dotenv import load_dotenv

load_dotenv()

classification_app = ClassificationApp()

classification_chain = classification_app.classification_chain

print(classification_chain.invoke({
    "input": "no me gusto para nada el servicio que ofrecen"
}))