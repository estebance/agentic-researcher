from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from services.config.config_model import BedrockKdbRetrieverParams

class KDBRetriever:

    def __init__(self, kdb_config_params: BedrockKdbRetrieverParams, provider='AWS'):
        self.kdb_config_params = kdb_config_params
        self.retriever = None
        if provider == 'AWS':
            self.retriever = self.retrieve_bedrock_kdb()
        else:
            raise Exception("provider not defined")

    def retrieve_bedrock_kdb(self):
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=self.kdb_config_params.kdb_id,
            region_name=self.kdb_config_params.kdb_region,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": self.kdb_config_params.kdb_max_number_of_results
                }
            },
        )
        return retriever