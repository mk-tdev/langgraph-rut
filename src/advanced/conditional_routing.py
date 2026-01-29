"""
Advanced Conditional Routing in LangGraph

This example demonstrates sophisticated conditional routing patterns including:
- Multi-way conditional routing based on complex state analysis
- Dynamic routing with multiple criteria
- Nested conditional logic
- Routing based on message content, sentiment, and intent classification
"""

import os
from typing import Literal, TypedDict, Annotated, Sequence, Union
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from pydantic import BaseModel, Field
import re
from langgraph_viz import visualize

# Initialize LLM
llm = ChatOllama(model="gpt-oss:120b-cloud")

class IntentType(str, Enum):
    QUESTION = "question"
    COMMAND = "command"
    CONVERSATION = "conversation"
    REQUEST = "request"

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class UrgencyLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class MessageAnalysis(BaseModel):
    """Analysis of user message for routing decisions"""
    intent: IntentType = Field(description="The intent type of the message")
    sentiment: SentimentType = Field(description="The sentiment of the message")
    urgency: UrgencyLevel = Field(description="The urgency level of the message")
    complexity_score: float = Field(description="Complexity score from 0.0 to 1.0")
    requires_research: bool = Field(description="Whether the message requires external research")
    topic_category: str = Field(description="The main topic category")

class AdvancedState(TypedDict):
    """Enhanced state with multiple analysis fields"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    analysis: MessageAnalysis
    routing_path: list[str]
    processing_stage: str
    metadata: dict

def analyze_message_complexity(message: str) -> float:
    """Calculate complexity score based on message characteristics"""
    complexity_factors = {
        'length': min(len(message.split()) / 50, 1.0),  # Normalize to 0-1
        'question_marks': message.count('?') * 0.1,
        'exclamation_marks': message.count('!') * 0.1,
        'technical_terms': len(re.findall(r'\b(data|algorithm|API|database|system)\b', message, re.IGNORECASE)) * 0.2,
        'multiple_sentences': message.count('.') * 0.05
    }
    
    return min(sum(complexity_factors.values()), 1.0)

def classify_urgency(message: str) -> UrgencyLevel:
    """Classify message urgency based on keywords and patterns"""
    urgent_keywords = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'now']
    medium_keywords = ['soon', 'quickly', 'help', 'need', 'please']
    
    message_lower = message.lower()
    
    if any(keyword in message_lower for keyword in urgent_keywords):
        return UrgencyLevel.HIGH
    elif any(keyword in message_lower for keyword in medium_keywords):
        return UrgencyLevel.MEDIUM
    else:
        return UrgencyLevel.LOW

def extract_topic_category(message: str) -> str:
    """Extract the main topic category from the message"""
    categories = {
        'technical': ['code', 'programming', 'software', 'development', 'api', 'database'],
        'business': ['business', 'sales', 'marketing', 'revenue', 'customer'],
        'general': ['weather', 'news', 'general', 'information'],
        'personal': ['I', 'my', 'me', 'personal', 'opinion'],
        'help': ['help', 'assist', 'support', 'guidance', 'how to']
    }
    
    message_lower = message.lower()
    
    for category, keywords in categories.items():
        if any(keyword in message_lower for keyword in keywords):
            return category
    
    return 'general'

def comprehensive_message_analyzer(state: AdvancedState) -> AdvancedState:
    """
    Advanced message analysis that considers multiple dimensions
    """
    last_message = state["messages"][-1]
    message_content = last_message.content
    
    # Intent classification
    if message_content.endswith('?'):
        intent = IntentType.QUESTION
    elif any(command in message_content.lower() for command in ['please', 'can you', 'would you', 'help me']):
        intent = IntentType.REQUEST
    elif any(verb in message_content.lower() for verb in ['create', 'make', 'build', 'implement', 'develop']):
        intent = IntentType.COMMAND
    else:
        intent = IntentType.CONVERSATION
    
    # Sentiment analysis (simplified)
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'thanks', 'appreciate']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'wrong', 'error', 'problem', 'issue']
    
    message_lower = message_content.lower()
    positive_count = sum(1 for word in positive_words if word in message_lower)
    negative_count = sum(1 for word in negative_words if word in message_lower)
    
    if positive_count > negative_count:
        sentiment = SentimentType.POSITIVE
    elif negative_count > positive_count:
        sentiment = SentimentType.NEGATIVE
    else:
        sentiment = SentimentType.NEUTRAL
    
    # Create comprehensive analysis
    analysis = MessageAnalysis(
        intent=intent,
        sentiment=sentiment,
        urgency=classify_urgency(message_content),
        complexity_score=analyze_message_complexity(message_content),
        requires_research=any(word in message_lower for word in ['research', 'find', 'look up', 'search']),
        topic_category=extract_topic_category(message_content)
    )
    
    state["analysis"] = analysis
    state["processing_stage"] = "analyzed"
    
    print(f"Message Analysis:")
    print(f"  Intent: {intent}")
    print(f"  Sentiment: {sentiment}")
    print(f"  Urgency: {urgency}")
    print(f"  Complexity: {analysis.complexity_score:.2f}")
    print(f"  Requires Research: {analysis.requires_research}")
    print(f"  Topic: {analysis.topic_category}")
    
    return state

def advanced_router(state: AdvancedState) -> str:
    """
    Multi-criteria routing decision based on comprehensive analysis
    """
    analysis = state["analysis"]
    
    # High urgency routing
    if analysis.urgency == UrgencyLevel.HIGH:
        if analysis.intent == IntentType.REQUEST:
            return "urgent_handler"
        else:
            return "priority_queue"
    
    # Complex research routing
    if analysis.requires_research and analysis.complexity_score > 0.7:
        return "research_agent"
    
    # Simple question routing
    if analysis.intent == IntentType.QUESTION and analysis.complexity_score < 0.4:
        return "quick_answer"
    
    # Technical topic routing
    if analysis.topic_category == "technical":
        if analysis.complexity_score > 0.6:
            return "technical_expert"
        else:
            return "technical_assistant"
    
    # Business topic routing
    if analysis.topic_category == "business":
        return "business_analyst"
    
    # Help request routing
    if analysis.topic_category == "help":
        return "help_desk"
    
    # Sentiment-based routing
    if analysis.sentiment == SentimentType.NEGATIVE:
        return "support_specialist"
    
    # Default routing
    return "general_assistant"

def secondary_router(state: AdvancedState) -> str:
    """
    Secondary routing for more granular decisions
    """
    analysis = state["analysis"]
    current_path = state["routing_path"]
    
    # Route based on current path and analysis
    if "technical" in current_path:
        if analysis.complexity_score > 0.8:
            return "senior_developer"
        else:
            return "junior_developer"
    
    if "research" in current_path:
        if analysis.topic_category == "technical":
            return "technical_research"
        else:
            return "general_research"
    
    return "default_processor"

def urgent_handler(state: AdvancedState) -> AdvancedState:
    """Handle urgent messages with priority"""
    response = AIMessage(
        content="🚨 URGENT: Your message has been flagged as high priority. "
                "I'm processing it immediately and will provide a rapid response."
    )
    state["messages"].append(response)
    state["routing_path"].append("urgent_handler")
    state["processing_stage"] = "urgent_processed"
    return state

def research_agent(state: AdvancedState) -> AdvancedState:
    """Handle complex research requests"""
    response = AIMessage(
        content="🔍 I'll conduct comprehensive research on your topic. "
                "This may take a moment as I gather and analyze multiple sources."
    )
    state["messages"].append(response)
    state["routing_path"].append("research_agent")
    state["processing_stage"] = "research_initiated"
    return state

def technical_expert(state: AdvancedState) -> AdvancedState:
    """Handle complex technical questions"""
    response = AIMessage(
        content="💻 Technical Expert: I'll analyze your technical question in depth. "
                "Let me examine the architecture, code patterns, and best practices."
    )
    state["messages"].append(response)
    state["routing_path"].append("technical_expert")
    state["processing_stage"] = "technical_analysis"
    return state

def quick_answer(state: AdvancedState) -> AdvancedState:
    """Handle simple questions with quick responses"""
    response = AIMessage(
        content="⚡ Quick Answer: I'll provide a direct response to your question."
    )
    state["messages"].append(response)
    state["routing_path"].append("quick_answer")
    state["processing_stage"] = "quick_responded"
    return state

def business_analyst(state: AdvancedState) -> AdvancedState:
    """Handle business-related inquiries"""
    response = AIMessage(
        content="📊 Business Analyst: I'll analyze your business question from multiple perspectives "
                "including market trends, financial implications, and strategic considerations."
    )
    state["messages"].append(response)
    state["routing_path"].append("business_analyst")
    state["processing_stage"] = "business_analysis"
    return state

def help_desk(state: AdvancedState) -> AdvancedState:
    """Handle help and support requests"""
    response = AIMessage(
        content="🛠️ Help Desk: I'm here to help! Let me assist you with step-by-step guidance "
                "and practical solutions."
    )
    state["messages"].append(response)
    state["routing_path"].append("help_desk")
    state["processing_stage"] = "help_provided"
    return state

def support_specialist(state: AdvancedState) -> AdvancedState:
    """Handle negative sentiment with specialized support"""
    response = AIMessage(
        content="💝 Support Specialist: I understand you may be experiencing some frustration. "
                "I'm here to help resolve any issues and ensure you get the support you need."
    )
    state["messages"].append(response)
    state["routing_path"].append("support_specialist")
    state["processing_stage"] = "specialized_support"
    return state

def priority_queue(state: AdvancedState) -> AdvancedState:
    """Handle high priority but non-urgent messages"""
    response = AIMessage(
        content="⏰ Priority Queue: Your message has been placed in the priority queue "
                "and will be processed with high priority."
    )
    state["messages"].append(response)
    state["routing_path"].append("priority_queue")
    state["processing_stage"] = "priority_queued"
    return state

def technical_assistant(state: AdvancedState) -> AdvancedState:
    """Handle basic technical questions"""
    response = AIMessage(
        content="🔧 Technical Assistant: I'll help you with your technical question "
                "using clear explanations and practical examples."
    )
    state["messages"].append(response)
    state["routing_path"].append("technical_assistant")
    state["processing_stage"] = "technical_assistance"
    return state

def general_assistant(state: AdvancedState) -> AdvancedState:
    """Handle general inquiries"""
    response = AIMessage(
        content="🤖 General Assistant: I'll help you with your request using my general knowledge "
                "and problem-solving capabilities."
    )
    state["messages"].append(response)
    state["routing_path"].append("general_assistant")
    state["processing_stage"] = "general_processed"
    return state

def senior_developer(state: AdvancedState) -> AdvancedState:
    """Handle complex technical issues requiring senior expertise"""
    response = AIMessage(
        content="👨‍💻 Senior Developer: I'll provide expert-level technical analysis "
                "considering system architecture, performance, and scalability."
    )
    state["messages"].append(response)
    state["routing_path"].append("senior_developer")
    state["processing_stage"] = "senior_review"
    return state

def junior_developer(state: AdvancedState) -> AdvancedState:
    """Handle standard technical issues"""
    response = AIMessage(
        content="👨‍💻 Junior Developer: I'll help with your technical question "
                "providing clear explanations and code examples."
    )
    state["messages"].append(response)
    state["routing_path"].append("junior_developer")
    state["processing_stage"] = "junior_review"
    return state

# Build the advanced conditional routing graph
def create_advanced_routing_graph():
    """Create the advanced conditional routing workflow"""
    
    workflow = StateGraph(AdvancedState)
    
    # Add nodes
    workflow.add_node("analyzer", comprehensive_message_analyzer)
    workflow.add_node("urgent_handler", urgent_handler)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("technical_expert", technical_expert)
    workflow.add_node("quick_answer", quick_answer)
    workflow.add_node("business_analyst", business_analyst)
    workflow.add_node("help_desk", help_desk)
    workflow.add_node("support_specialist", support_specialist)
    workflow.add_node("priority_queue", priority_queue)
    workflow.add_node("technical_assistant", technical_assistant)
    workflow.add_node("general_assistant", general_assistant)
    workflow.add_node("senior_developer", senior_developer)
    workflow.add_node("junior_developer", junior_developer)
    
    # Add edges
    workflow.add_edge(START, "analyzer")
    
    # Primary conditional routing from analyzer
    workflow.add_conditional_edges(
        "analyzer",
        advanced_router,
        {
            "urgent_handler": "urgent_handler",
            "research_agent": "research_agent",
            "quick_answer": "quick_answer",
            "technical_expert": "technical_expert",
            "business_analyst": "business_analyst",
            "help_desk": "help_desk",
            "support_specialist": "support_specialist",
            "priority_queue": "priority_queue",
            "technical_assistant": "technical_assistant",
            "general_assistant": "general_assistant"
        }
    )
    
    # Secondary conditional routing from technical_expert
    workflow.add_conditional_edges(
        "technical_expert",
        secondary_router,
        {
            "senior_developer": "senior_developer",
            "junior_developer": "junior_developer",
            "default_processor": END
        }
    )
    
    # Secondary conditional routing from research_agent
    workflow.add_conditional_edges(
        "research_agent",
        secondary_router,
        {
            "technical_research": "senior_developer",
            "general_research": "general_assistant",
            "default_processor": END
        }
    )
    
    # Direct edges to END for simple handlers
    workflow.add_edge("urgent_handler", END)
    workflow.add_edge("quick_answer", END)
    workflow.add_edge("business_analyst", END)
    workflow.add_edge("help_desk", END)
    workflow.add_edge("support_specialist", END)
    workflow.add_edge("priority_queue", END)
    workflow.add_edge("technical_assistant", END)
    workflow.add_edge("general_assistant", END)
    workflow.add_edge("senior_developer", END)
    workflow.add_edge("junior_developer", END)
    
    return workflow.compile()

# Create and test the graph
if __name__ == "__main__":
    # Create the graph
    advanced_graph = create_advanced_routing_graph()
    
    # Display the graph structure
    try:
        display(Image(advanced_graph.get_graph().draw_mermaid_png()))
    except:
        print("Graph visualization not available")
    
    print("\n" + "="*80)
    print("ADVANCED CONDITIONAL ROUTING TESTS")
    print("="*80)
    
    # Test messages with different characteristics
    test_messages = [
        "URGENT! I need help with my database connection right now!",
        "Can you explain how to implement a microservices architecture?",
        "What's the weather like today?",
        "I'm having a terrible time with this API integration, it keeps failing!",
        "Please help me understand machine learning algorithms",
        "How can our business improve customer retention rates?",
        "What is Python?",
        "I need comprehensive research on quantum computing applications in cryptography"
    ]
    
    print("\n🚀 Starting tests with visualization...")
    
    # Use the context manager to run with visualization
    with visualize(advanced_graph) as viz_app:
        print("Running with visualization - Browser will open at http://localhost:8765")
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- Test {i} ---")
            print(f"Input: {message}")
            
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "routing_path": [],
                "processing_stage": "initial"
            }
            
            # IMPORTANT: Use viz_app, not advanced_graph
            result = viz_app.invoke(initial_state)
            
            print(f"Routing Path: {' -> '.join(result['routing_path'])}")
            print(f"Processing Stage: {result['processing_stage']}")
            print(f"Response: {result['messages'][-1].content}")
            print("-" * 60)
    
    print("\n" + "="*80)
    print("TESTS COMPLETED - Visualization server closed")
    print("="*80)
