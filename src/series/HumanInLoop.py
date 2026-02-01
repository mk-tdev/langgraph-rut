from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display

llm = ChatOllama(model="gpt-oss:120b-cloud")
checkpointer = MemorySaver()


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


graph_builder = StateGraph(State)


@tool
def get_weather(location: str) -> str:
    """Get the weather for a given location."""
    # Dummy implementation for illustration
    return f"The weather in {location} is sunny."


@tool
def broken_api(location: str) -> str:
    """API call to get current weather"""
    return f"Currently no weather data available for {location}."


tools = [get_weather, broken_api]
llm_tools_node = llm.bind_tools(tools)


def chat_with_tools(state: State):
    messages = state["messages"]
    response = llm_tools_node.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools=tools)

graph_builder.add_node("chat_with_tools", chat_with_tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chat_with_tools", tools_condition)
graph_builder.add_edge("tools", "chat_with_tools")

graph_builder.set_entry_point("chat_with_tools")

graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["tools"])

display(Image(graph.get_graph().draw_mermaid_png()))

config = {"configurable": {"thread_id": "1"}}
input_message = HumanMessage(content="Hello, I am mk.")

graph.invoke({"messages": [input_message]}, config=config)

graph.invoke(
    {"messages": [HumanMessage(content="What is my name?")]},
    config=config,
)

snapshot = graph.get_state(config)
snapshot.next

config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="What is the weather in Singapore?")
graph.invoke({"messages": [input_message]}, config=config)

snapshot2 = graph.get_state(config)
existing_messages = snapshot2.values["messages"][-1]
existing_messages.pretty_print()
snapshot2.next
graph.invoke(None, config=config)
