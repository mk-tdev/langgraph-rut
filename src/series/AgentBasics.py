import os
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from IPython.display import Image, display
from langgraph_viz import visualize
from operator import add
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()

checkpointer = MemorySaver()
api_key = os.getenv("OLLAMA_API_KEY")

# Initialize the LLM
llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.7)

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

print(get_weather.invoke({"location": "Singapore"}))
print(get_traffic.invoke({"location": "Singapore"}))

tools = [get_weather, get_traffic]
print(tools)
llm_with_tools = llm.bind_tools([get_weather, get_traffic])

result = llm_with_tools.invoke("hello")
print(result.content)

def call_llm(state: MessagesState):
  messages = state["messages"]
  response = llm_with_tools.invoke(messages)
  return {"messages": [response]}

def should_continue(state: MessagesState) -> str:
  messages = state["messages"]
  last_message = messages[-1]
  if last_message.tool_calls:
    return "tools"
  return END

workflow = StateGraph(MessagesState)
tool_node = ToolNode(tools)
workflow.add_node("call_llm", call_llm)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges(
  "call_llm",
  should_continue,
  {
    "tools": "tools",
    END: END,
  }
)
workflow.add_edge("tools", "call_llm")

app = workflow.compile()
display(Image(app.get_graph().draw_mermaid_png()))

messages = [
  HumanMessage(content="What is the weather in India today?")
]

result = app.invoke({"messages": messages})
print([msg.pretty_print() for msg in result["messages"]])

app.invoke({"messages": [
  HumanMessage("What would you recommend in there then?")
]})


## With Checkpointer

workflow_with_memory = StateGraph(MessagesState)
tool_node = ToolNode(tools)
workflow_with_memory.add_node("call_llm", call_llm)
workflow_with_memory.add_node("tools", tool_node)

workflow_with_memory.add_edge(START, "call_llm")
workflow_with_memory.add_conditional_edges(
  "call_llm",
  should_continue,
  {
    "tools": "tools",
    END: END,
  }
)
workflow_with_memory.add_edge("tools", "call_llm")

app_with_memory = workflow_with_memory.compile(checkpointer=checkpointer)
display(Image(app_with_memory.get_graph().draw_mermaid_png()))
config = {"configurable": {"thread_id": "1"}}

result = app_with_memory.invoke(
  {"messages": [HumanMessage(content="What is the weather in India today?")]}, 
  config=config
)
print([msg.pretty_print() for msg in result["messages"]])

result = app_with_memory.invoke({"messages": [
  HumanMessage("What would you recommend in there then?")
]}, config=config)
print([msg.pretty_print() for msg in result["messages"]])
