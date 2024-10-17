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

TODO