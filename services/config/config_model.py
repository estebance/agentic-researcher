from enum import Enum
from pydantic import BaseModel
from typing import Optional

class ModelProvider(Enum):
    AWS = 'AWS'
    GOOGLE = 'GOOGLE'
    ANTHROPIC = 'ANTHROPIC'

class CheckpointerAuthParams(BaseModel):
    username: str
    password: str
    ssl: bool

class CheckpointerParams(BaseModel):
    endpoint: str
    port: int
    db_number: Optional[int] = None
    auth_params: Optional[CheckpointerAuthParams] = None

class WebRetrieverParams(BaseModel):
    urls: list[str]
    is_advanced_search: bool
    max_number_of_resources: int

class BedrockKdbRetrieverParams(BaseModel):
    kdb_id: str
    kdb_max_number_of_results: int
    kdb_region: str

class ToolsParams(BaseModel):
    name: str
    description: str
    tool_schema: dict
    endpoint: str
    endpoint_config: dict

class WokerParams(BaseModel):
    id: str
    name: str
    task: str
    tools: list[ToolsParams]

class ResearcherWorkerParams(BaseModel):
    id: str
    task: str
    enabled: bool
    web_retriever: Optional[WebRetrieverParams] = None
    bedrock_kdb_retriever_params: BedrockKdbRetrieverParams

class ModelProviderParams(BaseModel):
    provider: ModelProvider
    llm_model_id: str
    temperature: float
    max_tokens: int
    provider_args: Optional[dict] = None

class ParametrizationAgent(BaseModel):
    llm_model_provider: ModelProviderParams
    checkpointer: CheckpointerParams
    workers: Optional[list[WokerParams]] = None
    researcher_worker: ResearcherWorkerParams