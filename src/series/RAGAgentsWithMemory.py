
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

template = """
  Answer the question based on only the following context and chat history. Especially take the latest question.

  Chat History: {chat_history}
  
  Context: {context}
  
  Question: {question}
  """
prompt = ChatPromptTemplate.from_template(template=template)

llm = ChatOllama(model="gpt-oss:120b-cloud")
rag_chain = prompt | llm

class AgentState(TypedDict):
  messages: list[BaseMessage]
  documents: list[Document]
  on_topic: str
  rephrased_question: str
  proceed_to_generate: bool
  rephrase_count: int
  question: HumanMessage

class GradeQuestion(BaseModel):
  """
  Boolean value to check whether a question is related to the restaurant Artic Vista
  """

  score: str = Field(
    description="Question is about restrauant? If yes -> 'Yes' if not -> 'No'"
  )

def question_rewrite(state: AgentState):
  print(f"Question rewrite: {state}")  

  state["documents"] = []
  state["on_topic"] = ""
  state["rephrased_question"] = ""
  state["proceed_to_generate"] = False
  state["rephrase_count"] = 0

  if "messages" not in state or state["messages"] is None:
    state["messages"] = []

  if state["question"] not in state["messages"]:
    state["messages"].append(state["question"])

  if len(state["messages"]) > 1:
    converation = state["messages"][:-1]
    current_question = state["question"].content

    messages = [
      SystemMessage(
        content="You are a helpful assistant that repharases the user's question to be a standalone query."
      )
    ]

    messages.extend(converation)
    messages.append(HumanMessage(content=current_question))
    rephrased_prompt = ChatPromptTemplate.from_messages(messages)
    prompt = rephrased_prompt.format()
    result = llm.invoke(prompt)
    print(f"rephrased_question {result.content.strip()}")
    state["rephrased_question"] = result.content.strip()
  else:
    print("rephrased_question {state['question'].content}")
    state["rephrased_question"] = state["question"].content

  return state

def question_classifier(state: AgentState):
  question = state["messages"][-1].content

  system_message = """
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

  human_message = HumanMessage(content=f"User question: {state['rephrased_question']}")

  # Create a prompt that asks for a simple Yes/No response
  grade_prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message
  ])
  
  structured_llm = llm.with_structured_output(GradeQuestion)

  # Chain the prompt with the LLM
  chain = grade_prompt | structured_llm
  
  # Get the response
  response = chain.invoke({})
  
  state["on_topic"] = response.score.strip()
  
  print(f"Question: {question}")
  print(f"Response: {response}")
  print(f"Is on topic: {state['on_topic']}")
  
  return state


def on_topic_router(state: AgentState):
  on_topic = state.get("on_topic", "").strip().lower()

  if on_topic == "yes":
    return "retrieve_documents"
  else:
    return "off_topic_response"

  
def retrieve_documents(state: AgentState): 
  """
  Retrieve documents from the vector store based on the user's question
  """
  documents = retriever.invoke(state['rephrased_question'])
  state["documents"] = documents
  return state

class GradeDocument(BaseModel):
  """
  Boolean value to check whether a document is relevant to the question
  """

  score: str = Field(
    description="Document is relevant to the question? If yes -> 'Yes' if not -> 'No'"
  )