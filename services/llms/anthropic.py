from langchain_anthropic import ChatAnthropic
# "anthropic.claude-3-sonnet-20240229-v1:0"
# model_args:
def retrieve_anthropic_chat(model_id, temperature):
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0)
    return llm