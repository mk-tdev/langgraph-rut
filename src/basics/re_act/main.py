import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")

# Initialize the LLM
llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.7)

# Test the LLM
result = llm.invoke("Hello, how are you?")
print("LLM Response:", result)

def check_weather(location: str) -> str:
    '''Return the weather forecast for the specified location.'''
    return f"It's always sunny in {location}"

graph = create_agent(
    model=llm,
    tools=[check_weather],
    system_prompt="You are a helpful assistant",
)
inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}

result = graph.invoke(inputs)
print(result)

for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)