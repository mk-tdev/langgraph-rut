
import os
from typing import Literal, TypedDict, Annotated, Sequence, Literal
from langchain_classic import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from langgraph_viz import visualize
from operator import add
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

try:
  from langchain_core.tools.retriever import create_retriever_tool
except ImportError:  # pragma: no cover
  try:
    from langchain.tools.retriever import create_retriever_tool
  except ImportError:  # pragma: no cover
    try:
      from langchain.tools import create_retriever_tool
    except ImportError:  # pragma: no cover
      from langchain_community.tools import create_retriever_tool

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Initialize Ollama with the granite-embedding model
embeddings = OllamaEmbeddings(model="granite-embedding:278m")

docs = [
    Document(
      page_content="Artic Vista is owned by Artic Group, a renowned chef with over 20 years of experience in the culinary industry.",
      metadata={"source": "owner.txt"}
    ),
    Document(
      page_content="Artic Vista is located in the heart of the city, offering a unique dining experience with its stunning architecture and ambiance.",
      metadata={"source": "location.txt"}
    ),
    Document(
      page_content="Artic Vista offers a diverse menu featuring both traditional and modern cuisine, with a focus on fresh, locally-sourced ingredients.",
      metadata={"source": "menu.txt"}
    ),
    Document(
      page_content="Artic Vista is known for its exceptional service and attention to detail, ensuring a memorable dining experience for every guest.",
      metadata={"source": "service.txt"}
    ),
    Document(
      page_content="Artic Vista is open from 11:00 AM to 10:00 PM, Monday to Sunday, with special events and private dining options available.",
      metadata={"source": "hours.txt"}
    ),
    Document(
      page_content="Artic Vista is committed to sustainability and environmental responsibility, using eco-friendly practices in all aspects of operations.",
      metadata={"source": "sustainability.txt"}
    ),
    Document(
      page_content="Artic Vista has received numerous awards and accolades for its exceptional cuisine, service, and overall dining experience.",
      metadata={"source": "awards.txt"}
    )
]

db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOllama(model="gpt-oss:120b-cloud")

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm

class AgentState(TypedDict):
  messages: list[BaseMessage]
  documents: list[Document]
  on_topic: str

class GradeQuestion(BaseModel):
  """
  Boolean value to check whether a question is related to the restaurant Artic Vista
  """

  score: str = Field(
    description="Question is about restrauant? If yes -> 'Yes' if not -> 'No'"
  )

def question_classifier(state: AgentState):
  question = state["messages"][-1].content

  system = """
    You are a question classifier. Given a question, determine whether it is related to the one of the following topics:

    1. Information about Artic Vista (the restaurant)
    2. Prices of dishes at Artic Vista (the restaurant)
    3. Opening hours of Artic Vista (the restaurant)

    If the question is about any of these topics, respond with a JSON object containing a 'score' field with value 'Yes'. 
    Otherwise, respond with a JSON object containing a 'score' field with value 'No'.
    
    Example responses:
    - For on-topic: {{"score": "Yes"}}
    - For off-topic: {{"score": "No"}}
  """

  # Create a prompt that asks for a simple Yes/No response
  grade_prompt = ChatPromptTemplate.from_messages([
      ("system", system),
      ("human", "User question: {question}")
  ])
  
  # Chain the prompt with the LLM
  chain = grade_prompt | llm
  
  # Get the response
  response = chain.invoke({"question": question})
  
  # Parse the response to get a simple Yes/No
  response_text = response.content.strip().lower()
  is_on_topic = "yes" in response_text
  
  print(f"Question: {question}")
  print(f"Response: {response_text}")
  print(f"Is on topic: {is_on_topic}")
  
  state["on_topic"] = "Yes" if is_on_topic else "No"
  return state

def on_topic_router(state: AgentState):
  on_topic = state["on_topic"]

  if on_topic.lower() == "yes":
    return "on_topic"
  else:
    return "off_topic"

def retrieve_documents(state: AgentState): 
  """
  Retrieve documents from the vector store based on the user's question
  """
  question = state["messages"][-1].content
  documents = retriever.invoke(question)
  state["documents"] = documents
  return state

def generate_answer(state: AgentState):
  """
  Generate an answer based on the retrieved documents
  """
  question = state["messages"][-1].content
  documents = state["documents"]
  context = format_docs(documents)
  answer = rag_chain.invoke({"context": context, "question": question})
  state["messages"].append(answer)
  return state

def off_topic_response(state: AgentState):
  """
  Respond to off-topic questions
  """
  state["messages"].append(
    AIMessage(content="I'm sorry, I can only answer questions about Artic Vista (the restaurant).")
  )
  return state

workflow = StateGraph(AgentState)

workflow.add_node("topic_decision", question_classifier)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("generate_answer", generate_answer)

workflow.add_conditional_edges(
  "topic_decision",
  on_topic_router,
  {
    "on_topic": "retrieve_documents",
    "off_topic": "off_topic_response"
  }
)

workflow.add_edge("retrieve_documents", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("off_topic_response", END)
workflow.set_entry_point("topic_decision")

graph = workflow.compile()
  
display(Image(graph.get_graph().draw_mermaid_png()))
config = {"configurable": {"thread_id": "1"}}

result = graph.invoke(
  input={"messages": [HumanMessage(content="What is the weather in India today?")]},
  config=config
)
print(result)
print(result["messages"][-1].content)

relevant_result = graph.invoke(
  input={"messages": [HumanMessage(content="What are the opening timings in Artic Vista?")]},
  config=config
)
print(relevant_result)
print(relevant_result["messages"][-1].content)


## use tools

retriever_tool = create_retriever_tool(
  retriever,
  "restaurant_info_retriever",
  "Retrieve information about Artic Vista (the restaurant) including owner, location, and other details."
)

@tool
def off_topic_response_tool():
  """
  Respond to off-topic questions
  """
  return "I'm sorry, I can only answer questions about Artic Vista (the restaurant)."

tools = [retriever_tool, off_topic_response_tool]

class AgentStateWithTools(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]

def agent(state):
  messages = state["messages"]
  model = ChatOllama(model="gpt-oss:120b-cloud")
  model_with_tools = model.bind_tools(tools)
  return {"messages": [model_with_tools.invoke(messages)]}

def should_continue(state: AgentStateWithTools) -> Literal["tools", END]:
  messages = state["messages"]
  last_message = messages[-1]
  if last_message.tool_calls:
    return "tools"
  return END

tools_workflow = StateGraph(AgentStateWithTools)
tools_workflow.add_node("agent", agent)
tools_workflow.add_node("tools", ToolNode(tools))
tools_workflow.add_edge(START, "agent")
tools_workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
tools_workflow.add_edge("tools", "agent")
tools_graph = tools_workflow.compile()
  
display(Image(tools_graph.get_graph().draw_mermaid_png()))
config = {"configurable": {"thread_id": "1"}}

print("\n🚀 Running RAG Agentic with Tools example with visualization...")

# Use the context manager to run with visualization
with visualize(tools_graph) as viz_app:
    print("Running with visualization - Browser will open at http://localhost:8765")
    
    # IMPORTANT: Use viz_app, not tools_graph
    viz_app.invoke(
      input={"messages": [HumanMessage(content="How will the weather be tomorrow?")]}
    )

    # IMPORTANT: Use viz_app, not tools_graph
    viz_app.invoke(
      input={"messages": [HumanMessage(content="When will Artic Vista be open?")]}
    )

print("\n✅ RAG Agentic with Tools demonstration completed - Visualization server closed")