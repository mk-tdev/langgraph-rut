from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from dataclasses import dataclass
from langgraph.prebuilt.chat_agent_executor import Runtime


@dataclass
class Context:
    """Pre-run context that replaces `config["configurable"]`.
    Additional attributes can be added later without touching the function signature.
    """

    language: str = "English"


llm = ChatOllama(model="gpt-oss:120b-cloud")


class ChatMessages(TypedDict):
    question: str
    answer: str
    llm_calls: int


class InputState(TypedDict):
    question: str


class PrivateState(TypedDict):
    llm_calls: int


class OutputState(TypedDict):
    answer: str


class OverAllState(InputState, PrivateState, OutputState):
    pass


def call_llm(state: OverAllState, runtime: Runtime[Context]):
    language = runtime.context.language
    system_message_content = f"You are a helpful assistant that responds in {language}."
    system_message = SystemMessage(content=system_message_content)
    messages = [system_message, HumanMessage(content=state["question"])]
    response = llm.invoke(messages)
    return {"answer": response}

workflow_ctx = StateGraph(
    state_schema=OverAllState,
    input_schema=InputState,
    output_schema=OutputState,
    context_schema=Context,
)
workflow_ctx.add_node("call_llm", call_llm)
workflow_ctx.add_edge(START, "call_llm")
workflow_ctx.add_edge("call_llm", END)
graph = workflow_ctx.compile()

context = Context(language="Hindi")
graph.invoke(input={"question": "What is the capital of Italy?"}, context=context)