from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from langgraph.graph.message import MessageGraph
load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")
llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.7)

graph = MessageGraph()