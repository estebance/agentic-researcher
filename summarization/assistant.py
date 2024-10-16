from langchain_core.messages import SystemMessage
from langchain_anthropic import ChatAnthropic
from state import State

model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
# Define the logic to call the model
def assistant(state: State):
    # Get summary if it exists
    summary = state.get("summary", "")
    if summary:
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"
        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": response}