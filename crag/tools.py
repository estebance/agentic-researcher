from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

# TODO check code to provide URLs and retrieve information only from them
def  retrieve_web_search_tool(config_params):
    web_search_tool = TavilySearchResults(k=config_params.max_number_of_resources)
    if config_params.urls:
        web_search_tool.include_domains = config_params.urls
    if config_params.is_advanced_search:
        web_search_tool.search_depth = "advanced"
    return web_search_tool


def retrieve_bedrock_kdb(config_params):
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=config_params.kdb_id,
        region_name=config_params.kdb_region,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": config_params.kdb_max_number_of_results}},
    )
    return retriever