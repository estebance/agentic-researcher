# Agentic-Researcher

Tests with LangGraph

# Concepts:

State: each graph execution creates a state that is passed between nodes in the graph as they execute, and each node updates this internal state with its return value after it executes

Corrective augmented generation (CRAG): Improve the robustness of generation keeping in mind that by retrieving from a static and limited corpus of data you could return sub-optimal documents. So you can use web searches as an extension for augmenting the retrieval results.


# Why CRAG ?

As mentioned in the paper “corrective Retrieval Augmented Generation” Large Language Models (LLMs) exhibit hallucinations since the accuracy of generated texts cannot be secured by the parametric knowledge they encapsulated.

# HOW TO

[RedisCheckPointer](https://langchain-ai.github.io/langgraph/how-tos/persistence_redis)
[CRAG](https://arxiv.org/pdf/2401.15884)


# ROADMAP

- [x] Project initialization
- [x] Integrate Tools (PG)
- [x] Agentic checkpointer
- [x] Integrate Endpoint
- [x] Test CRAG (Corrective RAG)
- [ ] Add ChatHistory to improve questio and stablish connection thru CRAG