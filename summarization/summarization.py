from nodes import summarize_conversation
from assistant import assistant
from langgraph.graph import StateGraph,  END, START
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from state import State
### EDGES ####
# Determine whether to end or summarize the conversation
def should_continue(state: State):
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    # Otherwise we can just end
    return END

### FOR THIS INITIAL TEST WE USE A MEMORY CHECKPOINTER ###

# GRAPH AND MEMORY #
# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", assistant)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

try:
    image = Image(graph.get_graph(xray=True).draw_mermaid_png())
    with open("net_image1.png", "wb") as fout:
        fout.write(image.data)
    # display(image)
except Exception:
    # This requires some extra dependencies and is optional
    pass

