"""
Parallel Execution in LangGraph

This example demonstrates advanced parallel execution patterns including:
- Parallel processing of multiple tasks
- Fan-out/fan-in patterns
- Concurrent API calls and data processing
- Parallel analysis and synthesis
- Dynamic parallel task creation
"""

import asyncio
import time
from typing import Literal, TypedDict, Annotated, Sequence, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from pydantic import BaseModel, Field
import random

# Initialize LLM
llm = ChatOllama(model="gpt-oss:120b-cloud")

class TaskResult(BaseModel):
    """Result of an individual parallel task"""
    task_id: str = Field(description="Unique identifier for the task")
    task_type: str = Field(description="Type of task performed")
    result: Any = Field(description="The result of the task")
    execution_time: float = Field(description="Time taken to execute the task")
    success: bool = Field(description="Whether the task completed successfully")
    error_message: str = Field(default="", description="Error message if task failed")

class ParallelState(TypedDict):
    """State for parallel execution workflows"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    parallel_tasks: List[Dict[str, Any]]
    task_results: List[TaskResult]
    synthesis_data: Dict[str, Any]
    execution_summary: Dict[str, Any]
    parallel_mode: str

def simulate_api_call(task_data: dict) -> TaskResult:
    """Simulate an external API call with variable delay"""
    task_id = task_data["task_id"]
    task_type = task_data["task_type"]
    
    # Simulate variable processing time
    delay = random.uniform(0.5, 2.0)
    time.sleep(delay)
    
    # Simulate occasional failures
    if random.random() < 0.1:  # 10% failure rate
        return TaskResult(
            task_id=task_id,
            task_type=task_type,
            result=None,
            execution_time=delay,
            success=False,
            error_message="Simulated API failure"
        )
    
    # Simulate different types of results based on task type
    if task_type == "sentiment_analysis":
        result = {"sentiment": random.choice(["positive", "negative", "neutral"]), "confidence": random.uniform(0.7, 0.95)}
    elif task_type == "entity_extraction":
        result = {"entities": [f"entity_{i}" for i in range(random.randint(1, 5))]}
    elif task_type == "topic_classification":
        result = {"topic": random.choice(["technology", "business", "politics", "sports"]), "score": random.uniform(0.6, 0.9)}
    elif task_type == "summarization":
        result = {"summary": f"Generated summary for {task_id}", "length": random.randint(50, 200)}
    else:
        result = {"data": f"Processed {task_type} for {task_id}"}
    
    return TaskResult(
        task_id=task_id,
        task_type=task_type,
        result=result,
        execution_time=delay,
        success=True
    )

def simulate_data_processing(task_data: dict) -> TaskResult:
    """Simulate data processing tasks"""
    task_id = task_data["task_id"]
    processing_type = task_data["processing_type"]
    data_size = task_data.get("data_size", 1000)
    
    # Simulate processing time based on data size
    delay = (data_size / 1000) * random.uniform(0.1, 0.5)
    time.sleep(delay)
    
    # Simulate processing results
    if processing_type == "aggregation":
        result = {"total": random.randint(1000, 10000), "average": random.uniform(10, 100)}
    elif processing_type == "transformation":
        result = {"transformed_records": data_size, "format": "json"}
    elif processing_type == "validation":
        result = {"valid_records": int(data_size * 0.9), "invalid_records": int(data_size * 0.1)}
    else:
        result = {"processed_items": data_size}
    
    return TaskResult(
        task_id=task_id,
        task_type=processing_type,
        result=result,
        execution_time=delay,
        success=True
    )

def simulate_ml_inference(task_data: dict) -> TaskResult:
    """Simulate machine learning inference tasks"""
    task_id = task_data["task_id"]
    model_type = task_data["model_type"]
    input_size = task_data.get("input_size", 100)
    
    # Simulate inference time
    delay = (input_size / 100) * random.uniform(0.2, 0.8)
    time.sleep(delay)
    
    # Simulate model predictions
    if model_type == "classification":
        result = {"predictions": [random.choice(["A", "B", "C"]) for _ in range(input_size)], "accuracy": random.uniform(0.8, 0.95)}
    elif model_type == "regression":
        result = {"predictions": [random.uniform(0, 100) for _ in range(input_size)], "mse": random.uniform(0.1, 0.5)}
    elif model_type == "clustering":
        result = {"clusters": random.randint(2, 8), "silhouette_score": random.uniform(0.3, 0.7)}
    else:
        result = {"output": f"ML inference result for {model_type}"}
    
    return TaskResult(
        task_id=task_id,
        task_type=model_type,
        result=result,
        execution_time=delay,
        success=True
    )

def create_parallel_tasks(state: ParallelState) -> ParallelState:
    """Create parallel tasks based on the input message"""
    message = state["messages"][-1].content
    parallel_mode = state.get("parallel_mode", "analysis")
    
    tasks = []
    
    if parallel_mode == "analysis":
        # Create multiple analysis tasks
        tasks = [
            {"task_id": f"sentiment_{int(time.time())}", "task_type": "sentiment_analysis"},
            {"task_id": f"entities_{int(time.time())}", "task_type": "entity_extraction"},
            {"task_id": f"topics_{int(time.time())}", "task_type": "topic_classification"},
            {"task_id": f"summary_{int(time.time())}", "task_type": "summarization"}
        ]
    elif parallel_mode == "data_processing":
        # Create data processing tasks
        tasks = [
            {"task_id": f"agg_{int(time.time())}", "processing_type": "aggregation", "data_size": 5000},
            {"task_id": f"trans_{int(time.time())}", "processing_type": "transformation", "data_size": 3000},
            {"task_id": f"val_{int(time.time())}", "processing_type": "validation", "data_size": 4000}
        ]
    elif parallel_mode == "ml_inference":
        # Create ML inference tasks
        tasks = [
            {"task_id": f"class_{int(time.time())}", "model_type": "classification", "input_size": 200},
            {"task_id": f"reg_{int(time.time())}", "model_type": "regression", "input_size": 150},
            {"task_id": f"cluster_{int(time.time())}", "model_type": "clustering", "input_size": 300}
        ]
    
    state["parallel_tasks"] = tasks
    print(f"Created {len(tasks)} parallel tasks for {parallel_mode}")
    
    return state

def execute_parallel_tasks(state: ParallelState) -> ParallelState:
    """Execute tasks in parallel using ThreadPoolExecutor"""
    tasks = state["parallel_tasks"]
    parallel_mode = state.get("parallel_mode", "analysis")
    
    results = []
    
    # Choose the appropriate execution function based on mode
    if parallel_mode == "analysis":
        execute_func = simulate_api_call
    elif parallel_mode == "data_processing":
        execute_func = simulate_data_processing
    elif parallel_mode == "ml_inference":
        execute_func = simulate_ml_inference
    else:
        execute_func = simulate_api_call
    
    # Execute tasks in parallel
    with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(execute_func, task): task for task in tasks}
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Task {result.task_id} completed in {result.execution_time:.2f}s")
            except Exception as exc:
                error_result = TaskResult(
                    task_id=task.get("task_id", "unknown"),
                    task_type=task.get("task_type", "unknown"),
                    result=None,
                    execution_time=0,
                    success=False,
                    error_message=str(exc)
                )
                results.append(error_result)
                print(f"Task {task.get('task_id', 'unknown')} failed: {exc}")
    
    state["task_results"] = results
    return state

def synthesize_results(state: ParallelState) -> ParallelState:
    """Synthesize results from parallel tasks"""
    results = state["task_results"]
    parallel_mode = state.get("parallel_mode", "analysis")
    
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    synthesis_data = {
        "total_tasks": len(results),
        "successful_tasks": len(successful_results),
        "failed_tasks": len(failed_results),
        "total_execution_time": sum(r.execution_time for r in results),
        "average_execution_time": sum(r.execution_time for r in results) / len(results) if results else 0,
        "parallel_efficiency": (sum(r.execution_time for r in results) / max(r.execution_time for r in results)) if results else 0
    }
    
    # Mode-specific synthesis
    if parallel_mode == "analysis":
        # Combine analysis results
        sentiment_results = [r.result for r in successful_results if r.task_type == "sentiment_analysis"]
        entity_results = [r.result for r in successful_results if r.task_type == "entity_extraction"]
        topic_results = [r.result for r in successful_results if r.task_type == "topic_classification"]
        summary_results = [r.result for r in successful_results if r.task_type == "summarization"]
        
        synthesis_data.update({
            "sentiment_analysis": sentiment_results,
            "entity_extraction": entity_results,
            "topic_classification": topic_results,
            "summarization": summary_results
        })
        
        response_content = f"""
📊 Parallel Analysis Complete:
- Sentiment Analysis: {len(sentiment_results)} results
- Entity Extraction: {len(entity_results)} results  
- Topic Classification: {len(topic_results)} results
- Summarization: {len(summary_results)} results
- Success Rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)
- Parallel Efficiency: {synthesis_data['parallel_efficiency']:.2f}x
        """
        
    elif parallel_mode == "data_processing":
        # Combine data processing results
        synthesis_data["processing_results"] = [r.result for r in successful_results]
        
        response_content = f"""
🔄 Parallel Data Processing Complete:
- Tasks Completed: {len(successful_results)}/{len(results)}
- Total Records Processed: {sum(r.result.get('processed_items', r.result.get('transformed_records', 0)) for r in successful_results)}
- Processing Time: {synthesis_data['total_execution_time']:.2f}s
- Parallel Speedup: {synthesis_data['parallel_efficiency']:.2f}x
        """
        
    elif parallel_mode == "ml_inference":
        # Combine ML inference results
        synthesis_data["inference_results"] = [r.result for r in successful_results]
        
        response_content = f"""
🤖 Parallel ML Inference Complete:
- Models Executed: {len(successful_results)}/{len(results)}
- Total Predictions: {sum(len(r.result.get('predictions', [])) for r in successful_results)}
- Inference Time: {synthesis_data['total_execution_time']:.2f}s
- Parallel Throughput: {synthesis_data['parallel_efficiency']:.2f}x
        """
    
    else:
        response_content = f"✅ Parallel execution completed with {len(successful_results)} successful tasks."
    
    state["synthesis_data"] = synthesis_data
    state["messages"].append(AIMessage(content=response_content.strip()))
    
    return state

def create_execution_summary(state: ParallelState) -> ParallelState:
    """Create a comprehensive execution summary"""
    synthesis_data = state["synthesis_data"]
    results = state["task_results"]
    
    summary = {
        "execution_stats": synthesis_data,
        "task_details": [
            {
                "task_id": r.task_id,
                "task_type": r.task_type,
                "success": r.success,
                "execution_time": r.execution_time,
                "error": r.error_message if not r.success else None
            }
            for r in results
        ],
        "performance_metrics": {
            "fastest_task": min(results, key=lambda x: x.execution_time).task_id if results else None,
            "slowest_task": max(results, key=lambda x: x.execution_time).task_id if results else None,
            "total_parallel_time": max(r.execution_time for r in results) if results else 0,
            "sequential_time_estimate": synthesis_data["total_execution_time"],
            "time_saved": synthesis_data["total_execution_time"] - max(r.execution_time for r in results) if results else 0
        }
    }
    
    state["execution_summary"] = summary
    
    # Add summary message
    response_content = f"""
📈 Execution Summary:
- Total Tasks: {synthesis_data['total_tasks']}
- Success Rate: {synthesis_data['successful_tasks']}/{synthesis_data['total_tasks']} ({synthesis_data['successful_tasks']/synthesis_data['total_tasks']*100:.1f}%)
- Parallel Time: {summary['performance_metrics']['total_parallel_time']:.2f}s
- Sequential Time Estimate: {summary['performance_metrics']['sequential_time_estimate']:.2f}s
- Time Saved: {summary['performance_metrics']['time_saved']:.2f}s
- Speedup: {synthesis_data['parallel_efficiency']:.2f}x
    """
    
    state["messages"].append(AIMessage(content=response_content.strip()))
    
    return state

def create_parallel_execution_graph():
    """Create the parallel execution workflow"""
    
    workflow = StateGraph(ParallelState)
    
    # Add nodes
    workflow.add_node("create_tasks", create_parallel_tasks)
    workflow.add_node("execute_tasks", execute_parallel_tasks)
    workflow.add_node("synthesize_results", synthesize_results)
    workflow.add_node("create_summary", create_execution_summary)
    
    # Add edges
    workflow.add_edge(START, "create_tasks")
    workflow.add_edge("create_tasks", "execute_tasks")
    workflow.add_edge("execute_tasks", "synthesize_results")
    workflow.add_edge("synthesize_results", "create_summary")
    workflow.add_edge("create_summary", END)
    
    return workflow.compile()

# Create and test the parallel execution graph
if __name__ == "__main__":
    # Create the graph
    parallel_graph = create_parallel_execution_graph()
    
    # Display the graph structure
    try:
        display(Image(parallel_graph.get_graph().draw_mermaid_png()))
    except:
        print("Graph visualization not available")
    
    # Test different parallel execution modes
    test_scenarios = [
        {"mode": "analysis", "message": "Analyze this text comprehensively"},
        {"mode": "data_processing", "message": "Process the dataset in parallel"},
        {"mode": "ml_inference", "message": "Run multiple ML models for inference"}
    ]
    
    print("\n" + "="*80)
    print("PARALLEL EXECUTION TESTS")
    print("="*80)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Test {i}: {scenario['mode'].upper()} ---")
        print(f"Input: {scenario['message']}")
        
        start_time = time.time()
        
        initial_state = {
            "messages": [HumanMessage(content=scenario['message'])],
            "parallel_mode": scenario['mode']
        }
        
        result = parallel_graph.invoke(initial_state)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Total Execution Time: {total_time:.2f}s")
        print(f"Final Response: {result['messages'][-1].content}")
        
        if i < len(test_scenarios):
            print("\n" + "="*60)
    
    print("\n" + "="*80)
    print("PARALLEL EXECUTION COMPLETED")
    print("="*80)
