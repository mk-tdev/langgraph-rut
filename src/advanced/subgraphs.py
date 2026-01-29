"""
Subgraphs and Modular Architecture in LangGraph

This example demonstrates advanced subgraph patterns including:
- Creating reusable subgraphs
- Nested subgraph architectures
- State management between parent and child graphs
- Dynamic subgraph selection
- Subgraph composition and chaining
"""

import json
from typing import Literal, TypedDict, Annotated, Sequence, List, Dict, Any, Optional, Union
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from pydantic import BaseModel, Field
import time

# Initialize LLM
llm = ChatOllama(model="gpt-oss:120b-cloud")

class ProcessingType(str, Enum):
    ANALYSIS = "analysis"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    ENRICHMENT = "enrichment"

class SubgraphType(str, Enum):
    DATA_PROCESSING = "data_processing"
    CONTENT_ANALYSIS = "content_analysis"
    QUALITY_CHECK = "quality_check"
    REPORT_GENERATION = "report_generation"

# Parent Graph State
class MainWorkflowState(TypedDict):
    """State for the main workflow that orchestrates subgraphs"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    input_data: Dict[str, Any]
    processing_pipeline: List[SubgraphType]
    subgraph_results: Dict[str, Any]
    final_output: Dict[str, Any]
    workflow_metadata: Dict[str, Any]

# Subgraph States
class DataProcessingState(TypedDict):
    """State for data processing subgraph"""
    raw_data: Dict[str, Any]
    processed_data: Dict[str, Any]
    processing_steps: List[str]
    quality_metrics: Dict[str, float]
    errors: List[str]

class ContentAnalysisState(TypedDict):
    """State for content analysis subgraph"""
    content: str
    analysis_results: Dict[str, Any]
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    topics: List[str]
    insights: List[str]

class QualityCheckState(TypedDict):
    """State for quality check subgraph"""
    data_to_check: Dict[str, Any]
    quality_score: float
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]
    passed_checks: List[str]
    failed_checks: List[str]

class ReportGenerationState(TypedDict):
    """State for report generation subgraph"""
    report_data: Dict[str, Any]
    report_sections: List[Dict[str, Any]]
    summary: str
    detailed_analysis: str
    recommendations: List[str]

# Data Processing Subgraph
def clean_data(state: DataProcessingState) -> DataProcessingState:
    """Clean and normalize raw data"""
    raw_data = state["raw_data"]
    
    # Simulate data cleaning
    cleaned_data = {
        "original_records": len(raw_data.get("records", [])),
        "cleaned_records": int(len(raw_data.get("records", [])) * 0.95),
        "removed_duplicates": int(len(raw_data.get("records", [])) * 0.05),
        "normalized_fields": ["name", "email", "date"],
        "data_quality": 0.92
    }
    
    state["processed_data"] = cleaned_data
    state["processing_steps"].append("data_cleaning")
    state["quality_metrics"]["cleanliness"] = 0.92
    
    print("🧹 Data cleaning completed")
    
    return state

def transform_data(state: DataProcessingState) -> DataProcessingState:
    """Transform data into required format"""
    processed_data = state["processed_data"]
    
    # Simulate data transformation
    transformed_data = {
        **processed_data,
        "transformed_format": "json",
        "schema_version": "2.0",
        "enriched_fields": ["category", "priority", "status"],
        "transformation_time": time.time()
    }
    
    state["processed_data"] = transformed_data
    state["processing_steps"].append("data_transformation")
    state["quality_metrics"]["transformation"] = 0.88
    
    print("🔄 Data transformation completed")
    
    return state

def validate_data(state: DataProcessingState) -> DataProcessingState:
    """Validate processed data"""
    processed_data = state["processed_data"]
    
    # Simulate data validation
    validation_results = {
        "schema_valid": True,
        "required_fields_present": True,
        "data_types_correct": True,
        "business_rules_satisfied": True,
        "validation_score": 0.95
    }
    
    state["processed_data"]["validation"] = validation_results
    state["processing_steps"].append("data_validation")
    state["quality_metrics"]["validation"] = 0.95
    
    print("✅ Data validation completed")
    
    return state

def create_data_processing_subgraph():
    """Create the data processing subgraph"""
    
    workflow = StateGraph(DataProcessingState)
    
    # Add nodes
    workflow.add_node("clean_data", clean_data)
    workflow.add_node("transform_data", transform_data)
    workflow.add_node("validate_data", validate_data)
    
    # Add edges
    workflow.add_edge(START, "clean_data")
    workflow.add_edge("clean_data", "transform_data")
    workflow.add_edge("transform_data", "validate_data")
    workflow.add_edge("validate_data", END)
    
    return workflow.compile()

# Content Analysis Subgraph
def analyze_sentiment(state: ContentAnalysisState) -> ContentAnalysisState:
    """Analyze sentiment of content"""
    content = state["content"]
    
    # Simulate sentiment analysis
    sentiment_result = {
        "overall_sentiment": "positive",
        "confidence": 0.87,
        "emotions": {
            "joy": 0.6,
            "trust": 0.3,
            "anticipation": 0.1
        },
        "sentiment_breakdown": {
            "positive": 0.7,
            "neutral": 0.2,
            "negative": 0.1
        }
    }
    
    state["sentiment"] = sentiment_result
    state["analysis_results"]["sentiment"] = sentiment_result
    
    print("😊 Sentiment analysis completed")
    
    return state

def extract_entities(state: ContentAnalysisState) -> ContentAnalysisState:
    """Extract entities from content"""
    content = state["content"]
    
    # Simulate entity extraction
    entities = [
        {"text": "LangGraph", "type": "TECHNOLOGY", "confidence": 0.95},
        {"text": "AI", "type": "CONCEPT", "confidence": 0.92},
        {"text": "workflow", "type": "PROCESS", "confidence": 0.88}
    ]
    
    state["entities"] = entities
    state["analysis_results"]["entities"] = entities
    
    print("🏷️ Entity extraction completed")
    
    return state

def identify_topics(state: ContentAnalysisState) -> ContentAnalysisState:
    """Identify main topics in content"""
    content = state["content"]
    
    # Simulate topic identification
    topics = ["artificial intelligence", "workflow automation", "data processing", "machine learning"]
    
    state["topics"] = topics
    state["analysis_results"]["topics"] = topics
    
    print("📋 Topic identification completed")
    
    return state

def generate_insights(state: ContentAnalysisState) -> ContentAnalysisState:
    """Generate insights from analysis"""
    analysis_results = state["analysis_results"]
    
    # Simulate insight generation
    insights = [
        "The content shows strong positive sentiment",
        "Key entities indicate a technical focus",
        "Topics suggest AI and automation themes",
        "Overall content quality is high"
    ]
    
    state["insights"] = insights
    state["analysis_results"]["insights"] = insights
    
    print("💡 Insight generation completed")
    
    return state

def create_content_analysis_subgraph():
    """Create the content analysis subgraph"""
    
    workflow = StateGraph(ContentAnalysisState)
    
    # Add nodes
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("identify_topics", identify_topics)
    workflow.add_node("generate_insights", generate_insights)
    
    # Add edges (can be parallel for some operations)
    workflow.add_edge(START, "analyze_sentiment")
    workflow.add_edge(START, "extract_entities")
    workflow.add_edge(START, "identify_topics")
    workflow.add_edge("analyze_sentiment", "generate_insights")
    workflow.add_edge("extract_entities", "generate_insights")
    workflow.add_edge("identify_topics", "generate_insights")
    workflow.add_edge("generate_insights", END)
    
    return workflow.compile()

# Quality Check Subgraph
def check_completeness(state: QualityCheckState) -> QualityCheckState:
    """Check data completeness"""
    data = state["data_to_check"]
    
    # Simulate completeness check
    completeness_score = 0.85
    missing_fields = ["optional_field_1", "optional_field_2"]
    
    state["quality_score"] = completeness_score
    state["passed_checks"].append("completeness_check")
    if missing_fields:
        state["issues_found"].append({
            "type": "missing_fields",
            "severity": "low",
            "description": f"Missing optional fields: {missing_fields}"
        })
    
    print("📊 Completeness check completed")
    
    return state

def check_accuracy(state: QualityCheckState) -> QualityCheckState:
    """Check data accuracy"""
    data = state["data_to_check"]
    
    # Simulate accuracy check
    accuracy_score = 0.92
    accuracy_issues = []
    
    state["quality_score"] = (state["quality_score"] + accuracy_score) / 2
    state["passed_checks"].append("accuracy_check")
    
    if accuracy_issues:
        state["issues_found"].extend(accuracy_issues)
    
    print("🎯 Accuracy check completed")
    
    return state

def check_consistency(state: QualityCheckState) -> QualityCheckState:
    """Check data consistency"""
    data = state["data_to_check"]
    
    # Simulate consistency check
    consistency_score = 0.89
    consistency_issues = [
        {
            "type": "format_inconsistency",
            "severity": "medium",
            "description": "Date format inconsistency in some records"
        }
    ]
    
    state["quality_score"] = (state["quality_score"] + consistency_score) / 2
    state["passed_checks"].append("consistency_check")
    state["issues_found"].extend(consistency_issues)
    
    print("⚖️ Consistency check completed")
    
    return state

def generate_quality_recommendations(state: QualityCheckState) -> QualityCheckState:
    """Generate quality improvement recommendations"""
    issues = state["issues_found"]
    
    recommendations = []
    
    for issue in issues:
        if issue["type"] == "missing_fields":
            recommendations.append("Consider adding default values for missing optional fields")
        elif issue["type"] == "format_inconsistency":
            recommendations.append("Standardize date format across all records")
    
    state["recommendations"] = recommendations
    
    print("📝 Quality recommendations generated")
    
    return state

def create_quality_check_subgraph():
    """Create the quality check subgraph"""
    
    workflow = StateGraph(QualityCheckState)
    
    # Add nodes
    workflow.add_node("check_completeness", check_completeness)
    workflow.add_node("check_accuracy", check_accuracy)
    workflow.add_node("check_consistency", check_consistency)
    workflow.add_node("generate_recommendations", generate_quality_recommendations)
    
    # Add edges (parallel quality checks)
    workflow.add_edge(START, "check_completeness")
    workflow.add_edge(START, "check_accuracy")
    workflow.add_edge(START, "check_consistency")
    workflow.add_edge("check_completeness", "generate_recommendations")
    workflow.add_edge("check_accuracy", "generate_recommendations")
    workflow.add_edge("check_consistency", "generate_recommendations")
    workflow.add_edge("generate_recommendations", END)
    
    return workflow.compile()

# Report Generation Subgraph
def create_executive_summary(state: ReportGenerationState) -> ReportGenerationState:
    """Create executive summary"""
    report_data = state["report_data"]
    
    # Simulate executive summary creation
    summary = f"""
Executive Summary:
- Total records processed: {report_data.get('total_records', 'N/A')}
- Overall quality score: {report_data.get('quality_score', 'N/A')}
- Key insights identified: {len(report_data.get('insights', []))}
- Recommendations generated: {len(report_data.get('recommendations', []))}
    """.strip()
    
    state["summary"] = summary
    state["report_sections"].append({
        "title": "Executive Summary",
        "content": summary,
        "type": "summary"
    })
    
    print("📄 Executive summary created")
    
    return state

def create_detailed_analysis(state: ReportGenerationState) -> ReportGenerationState:
    """Create detailed analysis section"""
    report_data = state["report_data"]
    
    # Simulate detailed analysis creation
    detailed_analysis = f"""
Detailed Analysis:

Data Processing Results:
- Records cleaned: {report_data.get('cleaned_records', 'N/A')}
- Transformation success: {report_data.get('transformation_success', 'N/A')}

Content Analysis:
- Sentiment: {report_data.get('sentiment', 'N/A')}
- Entities found: {len(report_data.get('entities', []))}
- Topics identified: {len(report_data.get('topics', []))}

Quality Assessment:
- Overall quality: {report_data.get('quality_score', 'N/A')}
- Issues found: {len(report_data.get('issues', []))}
- Checks passed: {len(report_data.get('passed_checks', []))}
    """.strip()
    
    state["detailed_analysis"] = detailed_analysis
    state["report_sections"].append({
        "title": "Detailed Analysis",
        "content": detailed_analysis,
        "type": "analysis"
    })
    
    print("📊 Detailed analysis created")
    
    return state

def create_recommendations_section(state: ReportGenerationState) -> ReportGenerationState:
    """Create recommendations section"""
    report_data = state["report_data"]
    
    # Get recommendations from report data or generate default ones
    recommendations = report_data.get("recommendations", [
        "Continue monitoring data quality",
        "Implement automated validation",
        "Enhance data processing pipeline"
    ])
    
    state["recommendations"] = recommendations
    state["report_sections"].append({
        "title": "Recommendations",
        "content": "\n".join(f"- {rec}" for rec in recommendations),
        "type": "recommendations"
    })
    
    print("💡 Recommendations section created")
    
    return state

def create_report_generation_subgraph():
    """Create the report generation subgraph"""
    
    workflow = StateGraph(ReportGenerationState)
    
    # Add nodes
    workflow.add_node("create_executive_summary", create_executive_summary)
    workflow.add_node("create_detailed_analysis", create_detailed_analysis)
    workflow.add_node("create_recommendations_section", create_recommendations_section)
    
    # Add edges
    workflow.add_edge(START, "create_executive_summary")
    workflow.add_edge("create_executive_summary", "create_detailed_analysis")
    workflow.add_edge("create_detailed_analysis", "create_recommendations_section")
    workflow.add_edge("create_recommendations_section", END)
    
    return workflow.compile()

# Main Workflow Functions
def initialize_main_workflow(state: MainWorkflowState) -> MainWorkflowState:
    """Initialize the main workflow"""
    
    # Define the processing pipeline
    pipeline = [
        SubgraphType.DATA_PROCESSING,
        SubgraphType.CONTENT_ANALYSIS,
        SubgraphType.QUALITY_CHECK,
        SubgraphType.REPORT_GENERATION
    ]
    
    state["processing_pipeline"] = pipeline
    state["subgraph_results"] = {}
    state["workflow_metadata"] = {
        "start_time": time.time(),
        "pipeline_steps": len(pipeline)
    }
    
    print(f"🚀 Initialized main workflow with {len(pipeline)} subgraphs")
    
    return state

def execute_data_processing_subgraph(state: MainWorkflowState) -> MainWorkflowState:
    """Execute the data processing subgraph"""
    
    # Create subgraph
    data_processing_graph = create_data_processing_subgraph()
    
    # Prepare initial state for subgraph
    subgraph_initial_state = {
        "raw_data": state["input_data"],
        "processed_data": {},
        "processing_steps": [],
        "quality_metrics": {},
        "errors": []
    }
    
    # Execute subgraph
    subgraph_result = data_processing_graph.invoke(subgraph_initial_state)
    
    # Store results
    state["subgraph_results"]["data_processing"] = subgraph_result
    state["messages"].append(AIMessage(
        content="✅ Data processing subgraph completed successfully"
    ))
    
    print("✅ Data processing subgraph execution completed")
    
    return state

def execute_content_analysis_subgraph(state: MainWorkflowState) -> MainWorkflowState:
    """Execute the content analysis subgraph"""
    
    # Create subgraph
    content_analysis_graph = create_content_analysis_subgraph()
    
    # Get content from previous subgraph results
    processed_data = state["subgraph_results"]["data_processing"]["processed_data"]
    content = f"Processed data with {processed_data.get('cleaned_records', 0)} records"
    
    # Prepare initial state for subgraph
    subgraph_initial_state = {
        "content": content,
        "analysis_results": {},
        "sentiment": {},
        "entities": [],
        "topics": [],
        "insights": []
    }
    
    # Execute subgraph
    subgraph_result = content_analysis_graph.invoke(subgraph_initial_state)
    
    # Store results
    state["subgraph_results"]["content_analysis"] = subgraph_result
    state["messages"].append(AIMessage(
        content="✅ Content analysis subgraph completed successfully"
    ))
    
    print("✅ Content analysis subgraph execution completed")
    
    return state

def execute_quality_check_subgraph(state: MainWorkflowState) -> MainWorkflowState:
    """Execute the quality check subgraph"""
    
    # Create subgraph
    quality_check_graph = create_quality_check_subgraph()
    
    # Combine data from previous subgraphs
    data_to_check = {
        "data_processing": state["subgraph_results"]["data_processing"]["processed_data"],
        "content_analysis": state["subgraph_results"]["content_analysis"]["analysis_results"]
    }
    
    # Prepare initial state for subgraph
    subgraph_initial_state = {
        "data_to_check": data_to_check,
        "quality_score": 0.0,
        "issues_found": [],
        "recommendations": [],
        "passed_checks": [],
        "failed_checks": []
    }
    
    # Execute subgraph
    subgraph_result = quality_check_graph.invoke(subgraph_initial_state)
    
    # Store results
    state["subgraph_results"]["quality_check"] = subgraph_result
    state["messages"].append(AIMessage(
        content="✅ Quality check subgraph completed successfully"
    ))
    
    print("✅ Quality check subgraph execution completed")
    
    return state

def execute_report_generation_subgraph(state: MainWorkflowState) -> MainWorkflowState:
    """Execute the report generation subgraph"""
    
    # Create subgraph
    report_generation_graph = create_report_generation_subgraph()
    
    # Combine all previous results
    report_data = {
        "total_records": state["subgraph_results"]["data_processing"]["processed_data"].get("cleaned_records", 0),
        "quality_score": state["subgraph_results"]["quality_check"]["quality_score"],
        "insights": state["subgraph_results"]["content_analysis"]["insights"],
        "recommendations": state["subgraph_results"]["quality_check"]["recommendations"],
        "issues": state["subgraph_results"]["quality_check"]["issues_found"],
        "passed_checks": state["subgraph_results"]["quality_check"]["passed_checks"],
        "sentiment": state["subgraph_results"]["content_analysis"]["sentiment"]["overall_sentiment"],
        "entities": state["subgraph_results"]["content_analysis"]["entities"],
        "topics": state["subgraph_results"]["content_analysis"]["topics"]
    }
    
    # Prepare initial state for subgraph
    subgraph_initial_state = {
        "report_data": report_data,
        "report_sections": [],
        "summary": "",
        "detailed_analysis": "",
        "recommendations": []
    }
    
    # Execute subgraph
    subgraph_result = report_generation_graph.invoke(subgraph_initial_state)
    
    # Store results
    state["subgraph_results"]["report_generation"] = subgraph_result
    state["final_output"] = {
        "report": subgraph_result,
        "metadata": state["workflow_metadata"],
        "all_results": state["subgraph_results"]
    }
    
    state["messages"].append(AIMessage(
        content="✅ Report generation subgraph completed successfully"
    ))
    
    print("✅ Report generation subgraph execution completed")
    
    return state

def determine_next_subgraph(state: MainWorkflowState) -> str:
    """Determine which subgraph to execute next"""
    pipeline = state["processing_pipeline"]
    completed_subgraphs = list(state["subgraph_results"].keys())
    
    # Map subgraph types to execution functions
    subgraph_mapping = {
        SubgraphType.DATA_PROCESSING: "execute_data_processing",
        SubgraphType.CONTENT_ANALYSIS: "execute_content_analysis",
        SubgraphType.QUALITY_CHECK: "execute_quality_check",
        SubgraphType.REPORT_GENERATION: "execute_report_generation"
    }
    
    # Find next subgraph to execute
    for subgraph_type in pipeline:
        if subgraph_type.value not in completed_subgraphs:
            return subgraph_mapping[subgraph_type]
    
    # All subgraphs completed
    return "complete_workflow"

def complete_main_workflow(state: MainWorkflowState) -> MainWorkflowState:
    """Complete the main workflow and provide final summary"""
    
    end_time = time.time()
    total_time = end_time - state["workflow_metadata"]["start_time"]
    
    final_summary = f"""
🎉 MAIN WORKFLOW COMPLETED

Execution Summary:
- Total Subgraphs: {len(state['processing_pipeline'])}
- Completed Subgraphs: {len(state['subgraph_results'])}
- Total Execution Time: {total_time:.2f}s
- Average Subgraph Time: {total_time/len(state['subgraph_results']):.2f}s

Subgraph Results:
{chr(10).join(f"- {name}: {len(result)} fields" for name, result in state['subgraph_results'].items())}

Final Report Generated:
- Sections: {len(state['final_output']['report']['report_sections'])}
- Summary Length: {len(state['final_output']['report']['summary'])} characters
- Recommendations: {len(state['final_output']['report']['recommendations'])}

Workflow Metadata:
{json.dumps(state['workflow_metadata'], indent=2)}
    """
    
    state["messages"].append(AIMessage(content=final_summary.strip()))
    
    return state

def create_main_workflow_graph():
    """Create the main workflow that orchestrates all subgraphs"""
    
    workflow = StateGraph(MainWorkflowState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_main_workflow)
    workflow.add_node("execute_data_processing", execute_data_processing_subgraph)
    workflow.add_node("execute_content_analysis", execute_content_analysis_subgraph)
    workflow.add_node("execute_quality_check", execute_quality_check_subgraph)
    workflow.add_node("execute_report_generation", execute_report_generation_subgraph)
    workflow.add_node("complete_workflow", complete_main_workflow)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    
    # Add conditional edges for dynamic subgraph execution
    workflow.add_conditional_edges(
        "initialize",
        determine_next_subgraph,
        {
            "execute_data_processing": "execute_data_processing",
            "execute_content_analysis": "execute_content_analysis",
            "execute_quality_check": "execute_quality_check",
            "execute_report_generation": "execute_report_generation",
            "complete_workflow": "complete_workflow"
        }
    )
    
    workflow.add_conditional_edges(
        "execute_data_processing",
        determine_next_subgraph,
        {
            "execute_content_analysis": "execute_content_analysis",
            "execute_quality_check": "execute_quality_check",
            "execute_report_generation": "execute_report_generation",
            "complete_workflow": "complete_workflow"
        }
    )
    
    workflow.add_conditional_edges(
        "execute_content_analysis",
        determine_next_subgraph,
        {
            "execute_quality_check": "execute_quality_check",
            "execute_report_generation": "execute_report_generation",
            "complete_workflow": "complete_workflow"
        }
    )
    
    workflow.add_conditional_edges(
        "execute_quality_check",
        determine_next_subgraph,
        {
            "execute_report_generation": "execute_report_generation",
            "complete_workflow": "complete_workflow"
        }
    )
    
    workflow.add_edge("execute_report_generation", "complete_workflow")
    workflow.add_edge("complete_workflow", END)
    
    return workflow.compile()

# Create and test the subgraph architecture
if __name__ == "__main__":
    # Create the main workflow graph
    main_workflow_graph = create_main_workflow_graph()
    
    # Display the graph structure
    try:
        display(Image(main_workflow_graph.get_graph().draw_mermaid_png()))
    except:
        print("Graph visualization not available")
    
    print("\n" + "="*80)
    print("SUBGRAPH ARCHITECTURE DEMONSTRATION")
    print("="*80)
    
    # Test the main workflow
    initial_state = {
        "messages": [HumanMessage(content="Execute the complete data processing pipeline")],
        "input_data": {
            "records": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
                {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
            ]
        }
    }
    
    print("\n🚀 Starting main workflow with subgraph orchestration...")
    
    # Run the workflow
    result = main_workflow_graph.invoke(initial_state)
    
    print("\n" + "="*80)
    print("MAIN WORKFLOW COMPLETED")
    print("="*80)
    
    # Display final messages
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(f"\n{message.content}")
    
    print(f"\nFinal Output Summary:")
    print(f"- Subgraphs Executed: {len(result['subgraph_results'])}")
    print(f"- Final Report Sections: {len(result['final_output']['report']['report_sections'])}")
    print(f"- Workflow Metadata: {result['workflow_metadata']}")
