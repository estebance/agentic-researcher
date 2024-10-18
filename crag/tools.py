from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

# TODO check code to provide URLs and retrieve information only from them
def  retrieve_web_search_tool(max_number_of_resources=3):
    web_search_tool = TavilySearchResults(k=max_number_of_resources)
    return web_search_tool


def retrieve_bedrock_kdb(kdb_id, kdb_region="us-east-1", kdb_number_of_results=3):
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kdb_id,
        region_name=kdb_region,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": kdb_number_of_results}},
    )
    return retriever