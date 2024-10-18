from langchain_core.messages import HumanMessage

# each agent node reports as HumanAgent
# name: refers to the agent role
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }

