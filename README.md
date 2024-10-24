# Deelab-Researcher

Deelab aproximation to create Agents capable of retrieve information as best as we can

## Concepts:

Corrective augmented generation (CRAG): Improve the robustness of generation keeping in mind that by retrieving from a static and limited corpus of data you could return sub-optimal documents. So you can use web searches as an extension for augmenting the retrieval results.

## Why CRAG ?

As mentioned in the paper “corrective Retrieval Augmented Generation” Large Language Models (LLMs) exhibit hallucinations since the accuracy of generated texts cannot be secured by the parametric knowledge they encapsulated.

## References

- [RedisCheckPointer](https://langchain-ai.github.io/langgraph/how-tos/persistence_redis)
- [CRAG](https://arxiv.org/pdf/2401.15884)

## Resources

- [LangGraph](https://www.langchain.com/langgraph) Orchestation and Agent Generation
- [FastApi](https://fastapi.tiangolo.com/) to expose an API
- [AWS-Bedrock](https://aws.amazon.com/es/bedrock/) for KDB
- [Redis](https://redis.io/) for the checkpointer
- [Poetry](https://python-poetry.org/) for dependency management
- [Anthropic](https://anthropic.com) LLM provider

# Install

1. The required python version is  ```python3.12```, you will find all the details by checking the ```pyproject.toml``` file.
2. Install poetry by following these [instructions](https://python-poetry.org/docs/)
3. After install poetry you can install the dependencies by running
    ```
        poetry install
    ````
4. Check the ```params.example.json```, because this is an initial version of this tool we only support AWS BEDROCK but you can manually use ANTHROPIC.
    ```
        {
            "provider": "AWS",
            "llm_model_id": "<YOUR BEDROCK MODEL ID>",
            "kdb_retriever_params": {
                "kdb_id": "<YOUR KDB ID>",
                "kdb_max_number_of_results": 3,
                "kdb_region": "us-east-1"
            },
            "web_retriever": {
                "enabled": false,
                "urls": [],
                "is_advanced_search": "true",
                "max_number_of_resources": 5
            },
            "checkpointer": {
                "endpoint": "127.0.0.1",
                "port": 6379,
                "db_number": 0,
                "auth_params": {
                    "username": "default",
                    "password": "<your password>",
                    "ssl": true
                }
            }
        }
    ```
    - Anthropic ref ```"llm_model_id": "claude-3-5-sonnet-20240620" ```

    Provide the values from your AWS Account
    - Bedrock with Anthropic ```"llm_model_id": "us.anthropic.claude-3-5-sonnet-20240620-v1:0"``` (cross regional inference)

5. Copy the content of ```params.example.json``` in ```params.json``` then provide the parameters

    - You can enable or disable de web search to do this just use the flag ```enabled=true/false``` available in the ```web_retriever```


## The checkpointer

The configuration of our checkpointer is pretty simple:

The following parameters are mandatory:

1. endpoint: Redis host
2. port: Redis port
3. db_number: integer
4. auth_params: is optional, provide these parameters for staging, prod environments.

Checkpointer configuration

```
    "checkpointer": {
        "endpoint": "127.0.0.1",
        "port": 6379,
        "db_number": 0,
        "auth_params": {
            "username": "default",
            "password": "<your password>",
            "ssl": true
        }
    }
```

## Environment

Copy the ```.env.example``` into a new file ```.env``` and replace the values

```
    ANTHROPIC_API_KEY="" (If you are using Anthropic)
    REDIS_ENDPOINT="127.0.0.1" (Your redis endpoint running)
    TAVILY_API_KEY="<your-tavily-api-key>"
    LANGSMITH_TRACING="true"
    LANGSMITH_API_KEY="<your langsmith api key>"
    LANGCHAIN_PROJECT="<your-ideal-project-name>"
    # AWS KDB and BEDROCK MODELS
    AWS_PROFILE="<your aws profile>"
    AWS_DEFAULT_REGION="us-east-1"
```