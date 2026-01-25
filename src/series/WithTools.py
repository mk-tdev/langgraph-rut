from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langgraph_viz import visualize
from operator import add
from pydantic import BaseModel
import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage

from dotenv import load_dotenv


load_dotenv()


@tool
def get_weather(location: str):
  """
  Get the weather for a given location.
  
  Args:
      location: The location to get the weather for.
  
  Returns:
      The weather for the given location.
  """
  return f"It's always sunny in {location}"

@tool
def get_traffic(location: str):
  """
  Get the traffic for a given location.
  
  Args:
      location: The location to get the traffic for.
  
  Returns:
      The traffic for the given location.
  """
  return f"Traffic is always smooth in {location}"


api_key = os.getenv("OLLAMA_API_KEY")

# Initialize the LLM
llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.7)
llm_with_tools = llm.bind_tools([get_weather, get_traffic])

# response = llm_with_tools.invoke("How will the weather be in Singapore today?")
# print(response)
# print(response.tool_calls)

messages = []
messages = [
  SystemMessage(
    "You are a helpful assistant, You have access to tools use them to provide answers"
  ),
  HumanMessage(
    "How will the weather be in Singapore today?"
  )
]
llm_ouput = llm_with_tools.invoke(messages)

messages.append(llm_ouput)
print('messages', messages)

tool_mapping = {
    "get_weather": get_weather,
    "get_traffic": get_traffic
}

for tool_call in llm_ouput.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_function = tool_mapping[tool_name]
    tool_result = tool_function.invoke(tool_args)
    messages.append(ToolMessage(tool_result, tool_call_id=tool_call["id"]))

print(messages)

llm_ouput = llm_with_tools.invoke(messages)
print(llm_ouput)
messages.append(llm_ouput)

