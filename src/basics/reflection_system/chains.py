from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")
llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.7)

# For the main prompt that generates tweets
tweet_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content=(
            "You are a twitter techie influencer assistant tasked with writing excellent tweets. "
            "Generate the best twitter post possible for the user's request. "
            "If user provides critique, respond with a revised version of the tweet."
        )
    ),
    ("human", "{input}")
])

# For the reflection prompt that critiques tweets
reflection_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content=(
            "You are a virtual twitter influencer grading a tweet. "
            "Generate critique and recommendations for the user's tweet. "
            "Always provide detailed recommendations, including feedback on length, style, and content."
        )
    ),
    ("human", "{tweet}")
])

# Create chains
tweet_chain = tweet_prompt | llm
reflection_chain = reflection_prompt | llm

