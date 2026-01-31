from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig


llm = ChatOllama(model="gpt-oss:120b-cloud")


class ChatMessages(TypedDict):
    question: str
    answer: str
    llm_calls: int


def call_llm(state: ChatMessages):
    question = state["question"]
    llm_calls = state.get("llm_calls", 0)

    state["llm_calls"] = llm_calls + 1
    print(f"LLM Call #{state['llm_calls']}")
    response = llm.invoke(input=question)
    state["answer"] = response.content
    return state


workflow = StateGraph(ChatMessages)
workflow.add_node("call_llm", call_llm)
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)

graph = workflow.compile()

graph.invoke(input={"question": "What is the capital of France?"})

class InputState(TypedDict):
    question: str
    
class PrivateState(TypedDict):
    llm_calls: int

class OutputState(TypedDict):
    answer: str
    
class OverAllState(InputState, PrivateState, OutputState):
    pass
  
workflow = StateGraph(OverAllState, input=InputState, output=OutputState)
workflow.add_node("call_llm", call_llm)
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)
graph = workflow.compile()
graph.invoke(input={"question": "What is the capital of Germany?"})


def call_llm(state: ChatMessages):
    question = state["question"]
    llm_calls = state.get("llm_calls", 0)

    state["llm_calls"] = llm_calls + 1
    print(f"LLM Call #{state['llm_calls']}")
    response = llm.invoke(input=question)
    state["answer"] = response.content
    return state

