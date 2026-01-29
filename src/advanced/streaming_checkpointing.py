"""
Streaming and Checkpointing in LangGraph

This example demonstrates advanced streaming and checkpointing patterns including:
- Real-time streaming of graph execution
- State persistence and recovery
- Checkpoint management and rollback
- Streaming with intermediate results
- Resumable workflows with state restoration
"""

import asyncio
import json
import time
from typing import Literal, TypedDict, Annotated, Sequence, List, Dict, Any, Optional, AsyncGenerator
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from IPython.display import Image, display
from pydantic import BaseModel, Field
import threading
import queue
from datetime import datetime

# Initialize LLM
llm = ChatOllama(model="gpt-oss:120b-cloud")

class StreamEvent(str, Enum):
    NODE_START = "node_start"
    NODE_END = "node_end"
    INTERMEDIATE_RESULT = "intermediate_result"
    ERROR = "error"
    CHECKPOINT = "checkpoint"
    STREAM_UPDATE = "stream_update"

class CheckpointType(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    ERROR_RECOVERY = "error_recovery"
    MILESTONE = "milestone"

class StreamData(BaseModel):
    """Model for streaming data"""
    event_type: StreamEvent = Field(description="Type of stream event")
    node_name: str = Field(description="Name of the node generating the event")
    timestamp: float = Field(description="Timestamp of the event")
    data: Dict[str, Any] = Field(description="Event data")
    progress: Optional[float] = Field(description="Progress percentage (0-1)")

class CheckpointInfo(BaseModel):
    """Model for checkpoint information"""
    checkpoint_id: str = Field(description="Unique checkpoint identifier")
    checkpoint_type: CheckpointType = Field(description="Type of checkpoint")
    timestamp: float = Field(description="Checkpoint creation time")
    state_snapshot: Dict[str, Any] = Field(description="Complete state snapshot")
    metadata: Dict[str, Any] = Field(description="Checkpoint metadata")

class StreamingCheckpointState(TypedDict):
    """State for streaming and checkpointing workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_node: str
    processing_steps: List[str]
    stream_events: List[StreamData]
    checkpoints: List[CheckpointInfo]
    progress: float
    error_history: List[Dict[str, Any]]
    recovery_attempts: int
    streaming_enabled: bool
    checkpoint_frequency: int

class StreamManager:
    """Manages streaming events and checkpoints"""
    
    def __init__(self):
        self.event_queue = queue.Queue()
        self.checkpoint_storage = {}
        self.subscribers = []
        self.streaming_active = False
    
    def emit_event(self, event: StreamData):
        """Emit a streaming event"""
        if self.streaming_active:
            self.event_queue.put(event)
            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber(event)
                except Exception as e:
                    print(f"Error notifying subscriber: {e}")
    
    def create_checkpoint(self, state: StreamingCheckpointState, checkpoint_type: CheckpointType = CheckpointType.AUTO) -> str:
        """Create a checkpoint of the current state"""
        checkpoint_id = f"ckpt_{int(time.time())}_{len(self.checkpoint_storage)}"
        
        checkpoint = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            timestamp=time.time(),
            state_snapshot=state.copy(),
            metadata={
                "current_node": state["current_node"],
                "progress": state["progress"],
                "processing_steps": state["processing_steps"]
            }
        )
        
        self.checkpoint_storage[checkpoint_id] = checkpoint
        
        # Emit checkpoint event
        event = StreamData(
            event_type=StreamEvent.CHECKPOINT,
            node_name="checkpoint_manager",
            timestamp=time.time(),
            data={"checkpoint_id": checkpoint_id, "type": checkpoint_type.value},
            progress=state.get("progress", 0.0)
        )
        self.emit_event(event)
        
        return checkpoint_id
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Restore state from a checkpoint"""
        checkpoint = self.checkpoint_storage.get(checkpoint_id)
        if checkpoint:
            return checkpoint.state_snapshot
        return None
    
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all available checkpoints"""
        return list(self.checkpoint_storage.values())
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint"""
        if checkpoint_id in self.checkpoint_storage:
            del self.checkpoint_storage[checkpoint_id]
            return True
        return False
    
    def start_streaming(self):
        """Start streaming events"""
        self.streaming_active = True
    
    def stop_streaming(self):
        """Stop streaming events"""
        self.streaming_active = False
    
    def subscribe(self, callback):
        """Subscribe to streaming events"""
        self.subscribers.append(callback)
    
    def get_events(self) -> List[StreamData]:
        """Get all events from the queue"""
        events = []
        while not self.event_queue.empty():
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events

# Global stream manager instance
stream_manager = StreamManager()

def stream_event_printer(event: StreamData):
    """Print streaming events to console"""
    timestamp_str = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp_str}] {event.event_type.value.upper()} - {event.node_name}")
    if event.data:
        print(f"  Data: {json.dumps(event.data, indent=2)}")
    if event.progress is not None:
        print(f"  Progress: {event.progress*100:.1f}%")
    print()

# Subscribe the printer to stream events
stream_manager.subscribe(stream_event_printer)

def data_ingestion(state: StreamingCheckpointState) -> StreamingCheckpointState:
    """Ingest and validate input data"""
    node_name = "data_ingestion"
    state["current_node"] = node_name
    
    # Emit node start event
    start_event = StreamData(
        event_type=StreamEvent.NODE_START,
        node_name=node_name,
        timestamp=time.time(),
        data={"message": "Starting data ingestion"},
        progress=0.0
    )
    stream_manager.emit_event(start_event)
    
    # Simulate data ingestion with progress updates
    total_records = 1000
    processed_records = 0
    
    for i in range(5):
        time.sleep(0.5)  # Simulate processing time
        processed_records += (total_records / 5)
        progress = processed_records / total_records
        
        # Emit intermediate result
        intermediate_event = StreamData(
            event_type=StreamEvent.INTERMEDIATE_RESULT,
            node_name=node_name,
            timestamp=time.time(),
            data={"processed_records": int(processed_records), "total_records": total_records},
            progress=progress * 0.2  # This node contributes 20% to total progress
        )
        stream_manager.emit_event(intermediate_event)
    
    # Complete data ingestion
    ingestion_result = {
        "total_records": total_records,
        "valid_records": int(total_records * 0.95),
        "invalid_records": int(total_records * 0.05),
        "data_quality_score": 0.95
    }
    
    state["processing_steps"].append(node_name)
    state["progress"] = 0.2
    
    # Emit node end event
    end_event = StreamData(
        event_type=StreamEvent.NODE_END,
        node_name=node_name,
        timestamp=time.time(),
        data={"result": ingestion_result},
        progress=0.2
    )
    stream_manager.emit_event(end_event)
    
    # Create automatic checkpoint
    stream_manager.create_checkpoint(state, CheckpointType.AUTO)
    
    print(f"✅ Data ingestion completed: {ingestion_result}")
    
    return state

def data_transformation(state: StreamingCheckpointState) -> StreamingCheckpointState:
    """Transform and process the ingested data"""
    node_name = "data_transformation"
    state["current_node"] = node_name
    
    # Emit node start event
    start_event = StreamData(
        event_type=StreamEvent.NODE_START,
        node_name=node_name,
        timestamp=time.time(),
        data={"message": "Starting data transformation"},
        progress=state["progress"]
    )
    stream_manager.emit_event(start_event)
    
    # Simulate transformation steps
    transformation_steps = ["normalization", "enrichment", "validation", "formatting"]
    
    for i, step in enumerate(transformation_steps):
        time.sleep(0.7)  # Simulate processing time
        step_progress = (i + 1) / len(transformation_steps)
        total_progress = state["progress"] + (step_progress * 0.3)  # This node contributes 30% to total progress
        
        # Emit intermediate result
        intermediate_event = StreamData(
            event_type=StreamEvent.INTERMEDIATE_RESULT,
            node_name=node_name,
            timestamp=time.time(),
            data={"step": step, "step_progress": step_progress},
            progress=total_progress
        )
        stream_manager.emit_event(intermediate_event)
        
        # Create milestone checkpoint at 50% progress
        if total_progress >= 0.5 and len(state["checkpoints"]) == 1:
            stream_manager.create_checkpoint(state, CheckpointType.MILESTONE)
    
    transformation_result = {
        "transformation_steps_completed": len(transformation_steps),
        "output_format": "json",
        "enrichment_applied": True,
        "validation_passed": True
    }
    
    state["processing_steps"].append(node_name)
    state["progress"] = 0.5
    
    # Emit node end event
    end_event = StreamData(
        event_type=StreamEvent.NODE_END,
        node_name=node_name,
        timestamp=time.time(),
        data={"result": transformation_result},
        progress=0.5
    )
    stream_manager.emit_event(end_event)
    
    print(f"✅ Data transformation completed: {transformation_result}")
    
    return state

def data_analysis(state: StreamingCheckpointState) -> StreamingCheckpointState:
    """Analyze the transformed data"""
    node_name = "data_analysis"
    state["current_node"] = node_name
    
    # Emit node start event
    start_event = StreamData(
        event_type=StreamEvent.NODE_START,
        node_name=node_name,
        timestamp=time.time(),
        data={"message": "Starting data analysis"},
        progress=state["progress"]
    )
    stream_manager.emit_event(start_event)
    
    # Simulate analysis with potential for errors
    analysis_tasks = ["statistical_analysis", "pattern_detection", "anomaly_detection", "insight_generation"]
    
    for i, task in enumerate(analysis_tasks):
        time.sleep(0.6)  # Simulate processing time
        
        # Simulate occasional error (10% chance)
        if task == "anomaly_detection" and hasattr(state, '_simulate_error') and state['_simulate_error']:
            error_event = StreamData(
                event_type=StreamEvent.ERROR,
                node_name=node_name,
                timestamp=time.time(),
                data={"error": "Simulated analysis error", "task": task},
                progress=state["progress"]
            )
            stream_manager.emit_event(error_event)
            
            # Create error recovery checkpoint
            stream_manager.create_checkpoint(state, CheckpointType.ERROR_RECOVERY)
            
            state["error_history"].append({
                "node": node_name,
                "task": task,
                "error": "Simulated analysis error",
                "timestamp": time.time()
            })
            
            # Continue with next task
            continue
        
        task_progress = (i + 1) / len(analysis_tasks)
        total_progress = state["progress"] + (task_progress * 0.3)  # This node contributes 30% to total progress
        
        # Emit intermediate result
        intermediate_event = StreamData(
            event_type=StreamEvent.INTERMEDIATE_RESULT,
            node_name=node_name,
            timestamp=time.time(),
            data={"task": task, "task_progress": task_progress},
            progress=total_progress
        )
        stream_manager.emit_event(intermediate_event)
    
    analysis_result = {
        "analysis_tasks_completed": len([t for t in analysis_tasks if t != "anomaly_detection" or not getattr(state, '_simulate_error', False)]),
        "insights_found": 15,
        "anomalies_detected": 3,
        "analysis_quality": 0.92
    }
    
    state["processing_steps"].append(node_name)
    state["progress"] = 0.8
    
    # Emit node end event
    end_event = StreamData(
        event_type=StreamEvent.NODE_END,
        node_name=node_name,
        timestamp=time.time(),
        data={"result": analysis_result},
        progress=0.8
    )
    stream_manager.emit_event(end_event)
    
    print(f"✅ Data analysis completed: {analysis_result}")
    
    return state

def report_generation(state: StreamingCheckpointState) -> StreamingCheckpointState:
    """Generate final report"""
    node_name = "report_generation"
    state["current_node"] = node_name
    
    # Emit node start event
    start_event = StreamData(
        event_type=StreamEvent.NODE_START,
        node_name=node_name,
        timestamp=time.time(),
        data={"message": "Starting report generation"},
        progress=state["progress"]
    )
    stream_manager.emit_event(start_event)
    
    # Simulate report generation
    report_sections = ["executive_summary", "detailed_analysis", "recommendations", "appendix"]
    
    for i, section in enumerate(report_sections):
        time.sleep(0.4)  # Simulate processing time
        section_progress = (i + 1) / len(report_sections)
        total_progress = state["progress"] + (section_progress * 0.2)  # This node contributes 20% to total progress
        
        # Emit intermediate result
        intermediate_event = StreamData(
            event_type=StreamEvent.INTERMEDIATE_RESULT,
            node_name=node_name,
            timestamp=time.time(),
            data={"section": section, "section_progress": section_progress},
            progress=total_progress
        )
        stream_manager.emit_event(intermediate_event)
    
    report_result = {
        "sections_generated": len(report_sections),
        "total_pages": 25,
        "recommendations_count": 8,
        "report_quality": 0.96
    }
    
    state["processing_steps"].append(node_name)
    state["progress"] = 1.0
    
    # Emit node end event
    end_event = StreamData(
        event_type=StreamEvent.NODE_END,
        node_name=node_name,
        timestamp=time.time(),
        data={"result": report_result},
        progress=1.0
    )
    stream_manager.emit_event(end_event)
    
    # Create final manual checkpoint
    stream_manager.create_checkpoint(state, CheckpointType.MANUAL)
    
    print(f"✅ Report generation completed: {report_result}")
    
    return state

def create_streaming_checkpoint_graph():
    """Create the streaming and checkpointing workflow"""
    
    workflow = StateGraph(StreamingCheckpointState)
    
    # Add nodes
    workflow.add_node("data_ingestion", data_ingestion)
    workflow.add_node("data_transformation", data_transformation)
    workflow.add_node("data_analysis", data_analysis)
    workflow.add_node("report_generation", report_generation)
    
    # Add edges
    workflow.add_edge(START, "data_ingestion")
    workflow.add_edge("data_ingestion", "data_transformation")
    workflow.add_edge("data_transformation", "data_analysis")
    workflow.add_edge("data_analysis", "report_generation")
    workflow.add_edge("report_generation", END)
    
    # Add memory checkpointing
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

async def stream_execution_async(graph, initial_state: StreamingCheckpointState, config: Dict[str, Any]) -> AsyncGenerator[StreamData, None]:
    """Stream graph execution asynchronously"""
    
    # Start streaming
    stream_manager.start_streaming()
    
    # Create a custom event collector
    event_collector = queue.Queue()
    
    def event_collector_callback(event: StreamData):
        event_collector.put(event)
    
    stream_manager.subscribe(event_collector_callback)
    
    # Run the graph in a separate thread
    result_container = {}
    error_container = {}
    
    def run_graph():
        try:
            result = graph.invoke(initial_state, config=config)
            result_container["result"] = result
        except Exception as e:
            error_container["error"] = e
    
    graph_thread = threading.Thread(target=run_graph)
    graph_thread.start()
    
    # Stream events while graph is running
    while graph_thread.is_alive() or not event_collector.empty():
        try:
            event = event_collector.get(timeout=0.1)
            yield event
        except queue.Empty:
            continue
    
    # Wait for graph completion
    graph_thread.join()
    
    # Stop streaming
    stream_manager.stop_streaming()
    
    # Check for errors
    if "error" in error_container:
        raise error_container["error"]
    
    # Return final result
    if "result" in result_container:
        yield StreamData(
            event_type=StreamEvent.STREAM_UPDATE,
            node_name="execution_complete",
            timestamp=time.time(),
            data={"final_result": result_container["result"]},
            progress=1.0
        )

def demonstrate_checkpoint_recovery():
    """Demonstrate checkpoint recovery functionality"""
    print("\n" + "="*80)
    print("CHECKPOINT RECOVERY DEMONSTRATION")
    print("="*80)
    
    # Create graph
    graph = create_streaming_checkpoint_graph()
    
    # Set up initial state with error simulation
    initial_state = {
        "messages": [HumanMessage(content="Process data with streaming and checkpointing")],
        "current_node": "",
        "processing_steps": [],
        "stream_events": [],
        "checkpoints": [],
        "progress": 0.0,
        "error_history": [],
        "recovery_attempts": 0,
        "streaming_enabled": True,
        "checkpoint_frequency": 2
    }
    
    # Simulate error during analysis
    initial_state['_simulate_error'] = True
    
    config = {"configurable": {"thread_id": "checkpoint-demo"}}
    
    print("\n🚀 Starting execution with simulated error...")
    
    # Run until error occurs
    try:
        result = graph.invoke(initial_state, config=config)
    except Exception as e:
        print(f"❌ Execution interrupted: {e}")
    
    # List available checkpoints
    checkpoints = stream_manager.list_checkpoints()
    print(f"\n📋 Available checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  - {cp.checkpoint_id}: {cp.checkpoint_type.value} at {cp.metadata.get('progress', 0)*100:.1f}% progress")
    
    # Find the best checkpoint to restore from
    best_checkpoint = max(checkpoints, key=lambda cp: cp.metadata.get('progress', 0))
    print(f"\n🔄 Restoring from checkpoint: {best_checkpoint.checkpoint_id}")
    
    # Restore state from checkpoint
    restored_state = stream_manager.restore_from_checkpoint(best_checkpoint.checkpoint_id)
    
    if restored_state:
        # Remove error simulation for recovery
        restored_state['_simulate_error'] = False
        restored_state['recovery_attempts'] += 1
        
        print(f"✅ State restored. Progress: {restored_state['progress']*100:.1f}%")
        print(f"📈 Recovery attempt #{restored_state['recovery_attempts']}")
        
        # Continue execution from restored state
        print("\n🚀 Continuing execution from restored state...")
        
        # Create new graph for recovery
        recovery_graph = create_streaming_checkpoint_graph()
        recovery_config = {"configurable": {"thread_id": "checkpoint-recovery"}}
        
        try:
            final_result = recovery_graph.invoke(restored_state, config=recovery_config)
            print(f"✅ Recovery successful! Final progress: {final_result['progress']*100:.1f}%")
        except Exception as e:
            print(f"❌ Recovery failed: {e}")

def demonstrate_streaming_output():
    """Demonstrate real-time streaming output"""
    print("\n" + "="*80)
    print("REAL-TIME STREAMING DEMONSTRATION")
    print("="*80)
    
    # Create graph
    graph = create_streaming_checkpoint_graph()
    
    # Set up initial state
    initial_state = {
        "messages": [HumanMessage(content="Process data with real-time streaming")],
        "current_node": "",
        "processing_steps": [],
        "stream_events": [],
        "checkpoints": [],
        "progress": 0.0,
        "error_history": [],
        "recovery_attempts": 0,
        "streaming_enabled": True,
        "checkpoint_frequency": 2
    }
    
    config = {"configurable": {"thread_id": "streaming-demo"}}
    
    print("\n🚀 Starting streaming execution...")
    
    # Run with streaming
    async def run_streaming():
        event_count = 0
        async for event in stream_execution_async(graph, initial_state, config):
            event_count += 1
            # Events are already printed by the subscriber
            if event.event_type == StreamEvent.STREAM_UPDATE:
                print(f"\n🎉 Streaming completed with {event_count} events!")
                break
    
    # Run the async streaming
    asyncio.run(run_streaming())

# Main execution
if __name__ == "__main__":
    # Create the graph
    streaming_graph = create_streaming_checkpoint_graph()
    
    # Display the graph structure
    try:
        display(Image(streaming_graph.get_graph().draw_mermaid_png()))
    except:
        print("Graph visualization not available")
    
    print("\n" + "="*80)
    print("STREAMING AND CHECKPOINTING DEMONSTRATIONS")
    print("="*80)
    
    # Demonstrate streaming output
    demonstrate_streaming_output()
    
    # Demonstrate checkpoint recovery
    demonstrate_checkpoint_recovery()
    
    print("\n" + "="*80)
    print("DEMONSTRATIONS COMPLETED")
    print("="*80)
    
    # Final checkpoint summary
    final_checkpoints = stream_manager.list_checkpoints()
    print(f"\n📊 Final checkpoint summary:")
    print(f"- Total checkpoints created: {len(final_checkpoints)}")
    print(f"- Auto checkpoints: {len([cp for cp in final_checkpoints if cp.checkpoint_type == CheckpointType.AUTO])}")
    print(f"- Manual checkpoints: {len([cp for cp in final_checkpoints if cp.checkpoint_type == CheckpointType.MANUAL])}")
    print(f"- Milestone checkpoints: {len([cp for cp in final_checkpoints if cp.checkpoint_type == CheckpointType.MILESTONE])}")
    print(f"- Error recovery checkpoints: {len([cp for cp in final_checkpoints if cp.checkpoint_type == CheckpointType.ERROR_RECOVERY])}")
