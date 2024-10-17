from langchain_aws import ChatBedrock

# "anthropic.claude-3-sonnet-20240229-v1:0"
# model_args:
def retrieve_bedrock_chat(model_id, temperature, **model_kwargs):
    llm = ChatBedrock(
        model_id=model_id,
        model_kwargs={
            'temperature': temperature,
        } | model_kwargs
    )
    return llm