
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import retriever
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from IPython.display import Image, display
from langgraph_viz import visualize
from operator import add
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

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

retriever.invoke("What are the opening hours?")

template = """
  Answer the question based on only the following context:
  {context}
  
  Question: {question}
  """
prompt = ChatPromptTemplate.from_template(template=template)

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
  {
    "context": retriever | format_docs,
    "question": RunnablePassthrough(),
  }
  | prompt
  | ChatOllama(model="gpt-oss:120b-cloud")
  | StrOutputParser()
)

qa_chain.invoke("What are the opening hourse?")