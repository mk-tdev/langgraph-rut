"""
Custom State Management in LangGraph

This example demonstrates advanced custom state management patterns including:
- Custom state schemas and validation
- State transformation and normalization
- Multi-level state hierarchies
- State persistence and retrieval
- Custom state reducers and updaters
"""

import json
import time
from typing import Literal, TypedDict, Annotated, Sequence, List, Dict, Any, Optional, Union, Protocol
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass, field
import pickle
import hashlib
from abc import ABC, abstractmethod
from langgraph_viz import visualize

# Initialize LLM
llm = ChatOllama(model="gpt-oss:120b-cloud")

class StateType(str, Enum):
    WORKFLOW = "workflow"
    SESSION = "session"
    USER = "user"
    SYSTEM = "system"
    TEMPORARY = "temporary"

class StateOperation(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    TRANSFORM = "transform"

class StatePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Custom State Models
@dataclass
class StateMetadata:
    """Metadata for state entries"""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = field(default=1)
    state_type: StateType = StateType.WORKFLOW
    priority: StatePriority = StatePriority.MEDIUM
    tags: List[str] = field(default_factory=list)
    checksum: str = field(default="")
    access_count: int = field(default=0)
    size_bytes: int = field(default=0)

@dataclass
class StateSnapshot:
    """Snapshot of state at a point in time"""
    snapshot_id: str
    timestamp: float
    state_data: Dict[str, Any]
    metadata: StateMetadata
    parent_snapshot_id: Optional[str] = None
    diff_from_parent: Optional[Dict[str, Any]] = None

class StateValidator(ABC):
    """Abstract base class for state validators"""
    
    @abstractmethod
    def validate(self, state_data: Dict[str, Any]) -> bool:
        """Validate state data"""
        pass
    
    @abstractmethod
    def get_validation_errors(self, state_data: Dict[str, Any]) -> List[str]:
        """Get validation error messages"""
        pass

class WorkflowStateValidator(StateValidator):
    """Validator for workflow state"""
    
    def validate(self, state_data: Dict[str, Any]) -> bool:
        """Validate workflow state data"""
        required_fields = ["messages", "current_step", "status"]
        
        for field in required_fields:
            if field not in state_data:
                return False
        
        # Validate messages
        messages = state_data.get("messages", [])
        if not isinstance(messages, list):
            return False
        
        # Validate status
        status = state_data.get("status")
        if status not in ["pending", "running", "completed", "failed"]:
            return False
        
        return True
    
    def get_validation_errors(self, state_data: Dict[str, Any]) -> List[str]:
        """Get validation error messages"""
        errors = []
        
        required_fields = ["messages", "current_step", "status"]
        for field in required_fields:
            if field not in state_data:
                errors.append(f"Missing required field: {field}")
        
        messages = state_data.get("messages", [])
        if not isinstance(messages, list):
            errors.append("Messages must be a list")
        
        status = state_data.get("status")
        if status not in ["pending", "running", "completed", "failed"]:
            errors.append(f"Invalid status: {status}")
        
        return errors

class StateTransformer(ABC):
    """Abstract base class for state transformers"""
    
    @abstractmethod
    def transform(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform state data"""
        pass

class MessageNormalizer(StateTransformer):
    """Normalizes message objects in state"""
    
    def transform(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize messages to consistent format"""
        transformed_data = state_data.copy()
        
        messages = state_data.get("messages", [])
        normalized_messages = []
        
        for message in messages:
            if isinstance(message, dict):
                # Convert dict messages to BaseMessage
                if message.get("type") == "human":
                    normalized_messages.append(HumanMessage(content=message.get("content", "")))
                elif message.get("type") == "ai":
                    normalized_messages.append(AIMessage(content=message.get("content", "")))
                else:
                    normalized_messages.append(BaseMessage(content=message.get("content", "")))
            else:
                normalized_messages.append(message)
        
        transformed_data["messages"] = normalized_messages
        return transformed_data

class StateCompressor(StateTransformer):
    """Compresses large state data"""
    
    def transform(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress large data structures in state"""
        transformed_data = state_data.copy()
        
        # Compress large lists
        for key, value in state_data.items():
            if isinstance(value, list) and len(value) > 100:
                # Keep only first and last 50 items
                compressed_list = value[:50] + [{"...": f"... ({len(value) - 100} items omitted) ..."}] + value[-50:]
                transformed_data[key] = compressed_list
        
        # Compress large strings
        for key, value in state_data.items():
            if isinstance(value, str) and len(value) > 1000:
                # Truncate large strings
                transformed_data[key] = value[:500] + "... (truncated) ..." + value[-500:]
        
        return transformed_data

class CustomStateReducer:
    """Custom state reducer with advanced logic"""
    
    def __init__(self):
        self.validators = [WorkflowStateValidator()]
        self.transformers = [MessageNormalizer(), StateCompressor()]
    
    def reduce_state(self, current_state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce state with custom logic"""
        # Apply transformers to update
        transformed_update = update
        for transformer in self.transformers:
            transformed_update = transformer.transform(transformed_update)
        
        # Merge states
        new_state = current_state.copy()
        new_state.update(transformed_update)
        
        # Validate new state
        for validator in self.validators:
            if not validator.validate(new_state):
                errors = validator.get_validation_errors(new_state)
                print(f"⚠️ State validation errors: {errors}")
                # Optionally rollback or handle errors
                return current_state
        
        return new_state
    
    def merge_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple states"""
        merged_state = {}
        
        for state in states:
            merged_state = self.reduce_state(merged_state, state)
        
        return merged_state

class StateManager:
    """Advanced state management system"""
    
    def __init__(self):
        self.states: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, StateMetadata] = {}
        self.snapshots: Dict[str, StateSnapshot] = {}
        self.reducer = CustomStateReducer()
        self.state_history: List[Dict[str, Any]] = []
    
    def create_state(self, state_id: str, initial_data: Dict[str, Any], 
                    state_type: StateType = StateType.WORKFLOW,
                    priority: StatePriority = StatePriority.MEDIUM,
                    tags: List[str] = None) -> bool:
        """Create a new state"""
        if state_id in self.states:
            print(f"⚠️ State {state_id} already exists")
            return False
        
        # Apply initial transformations
        transformed_data = initial_data
        for transformer in self.reducer.transformers:
            transformed_data = transformer.transform(transformed_data)
        
        # Validate initial state
        for validator in self.reducer.validators:
            if not validator.validate(transformed_data):
                errors = validator.get_validation_errors(transformed_data)
                print(f"❌ State validation failed: {errors}")
                return False
        
        # Create metadata
        metadata = StateMetadata(
            created_at=time.time(),
            updated_at=time.time(),
            version=1,
            state_type=state_type,
            priority=priority,
            tags=tags or [],
            checksum=self._calculate_checksum(transformed_data),
            size_bytes=self._calculate_size(transformed_data)
        )
        
        self.states[state_id] = transformed_data
        self.metadata[state_id] = metadata
        
        print(f"✅ Created state {state_id}")
        return True
    
    def read_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Read a state"""
        if state_id not in self.states:
            print(f"❌ State {state_id} not found")
            return None
        
        # Update access count
        self.metadata[state_id].access_count += 1
        
        return self.states[state_id].copy()
    
    def update_state(self, state_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a state"""
        if state_id not in self.states:
            print(f"❌ State {state_id} not found")
            return False
        
        # Create snapshot before update
        self.create_snapshot(state_id, f"pre-update-{int(time.time())}")
        
        # Apply update with reducer
        current_state = self.states[state_id]
        new_state = self.reducer.reduce_state(current_state, update_data)
        
        # Update metadata
        metadata = self.metadata[state_id]
        metadata.updated_at = time.time()
        metadata.version += 1
        metadata.checksum = self._calculate_checksum(new_state)
        metadata.size_bytes = self._calculate_size(new_state)
        
        self.states[state_id] = new_state
        
        print(f"✅ Updated state {state_id} to version {metadata.version}")
        return True
    
    def delete_state(self, state_id: str) -> bool:
        """Delete a state"""
        if state_id not in self.states:
            print(f"❌ State {state_id} not found")
            return False
        
        del self.states[state_id]
        del self.metadata[state_id]
        
        # Clean up snapshots
        snapshots_to_delete = [sid for sid, snapshot in self.snapshots.items() 
                              if snapshot.state_data.get("state_id") == state_id]
        for sid in snapshots_to_delete:
            del self.snapshots[sid]
        
        print(f"✅ Deleted state {state_id}")
        return True
    
    def create_snapshot(self, state_id: str, snapshot_id: str = None) -> str:
        """Create a snapshot of current state"""
        if state_id not in self.states:
            print(f"❌ State {state_id} not found")
            return ""
        
        if snapshot_id is None:
            snapshot_id = f"snapshot_{state_id}_{int(time.time())}"
        
        # Find parent snapshot
        parent_snapshot_id = None
        state_snapshots = [s for s in self.snapshots.values() 
                          if s.state_data.get("state_id") == state_id]
        if state_snapshots:
            parent_snapshot = max(state_snapshots, key=lambda s: s.timestamp)
            parent_snapshot_id = parent_snapshot.snapshot_id
        
        # Calculate diff from parent
        diff_from_parent = None
        if parent_snapshot_id:
            parent_state = self.snapshots[parent_snapshot_id].state_data
            current_state = self.states[state_id]
            diff_from_parent = self._calculate_diff(parent_state, current_state)
        
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            state_data=self.states[state_id].copy(),
            metadata=self.metadata[state_id],
            parent_snapshot_id=parent_snapshot_id,
            diff_from_parent=diff_from_parent
        )
        
        self.snapshots[snapshot_id] = snapshot
        
        print(f"📸 Created snapshot {snapshot_id}")
        return snapshot_id
    
    def restore_snapshot(self, snapshot_id: str, target_state_id: str = None) -> bool:
        """Restore a state from snapshot"""
        if snapshot_id not in self.snapshots:
            print(f"❌ Snapshot {snapshot_id} not found")
            return False
        
        snapshot = self.snapshots[snapshot_id]
        
        if target_state_id is None:
            target_state_id = snapshot.state_data.get("state_id", f"restored_{int(time.time())}")
        
        # Restore state data
        self.states[target_state_id] = snapshot.state_data.copy()
        self.metadata[target_state_id] = snapshot.metadata
        
        print(f"🔄 Restored snapshot {snapshot_id} to state {target_state_id}")
        return True
    
    def list_states(self, state_type: StateType = None) -> List[str]:
        """List all states"""
        if state_type:
            return [sid for sid, meta in self.metadata.items() if meta.state_type == state_type]
        return list(self.states.keys())
    
    def get_state_info(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a state"""
        if state_id not in self.states:
            return None
        
        metadata = self.metadata[state_id]
        state_data = self.states[state_id]
        
        return {
            "state_id": state_id,
            "metadata": {
                "created_at": metadata.created_at,
                "updated_at": metadata.updated_at,
                "version": metadata.version,
                "state_type": metadata.state_type.value,
                "priority": metadata.priority.value,
                "tags": metadata.tags,
                "checksum": metadata.checksum,
                "access_count": metadata.access_count,
                "size_bytes": metadata.size_bytes
            },
            "data_keys": list(state_data.keys()),
            "snapshot_count": len([s for s in self.snapshots.values() 
                                 if s.state_data.get("state_id") == state_id])
        }
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for state data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _calculate_size(self, data: Dict[str, Any]) -> int:
        """Calculate size of state data in bytes"""
        return len(pickle.dumps(data))
    
    def _calculate_diff(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate difference between two state versions"""
        diff = {}
        
        all_keys = set(old_data.keys()) | set(new_data.keys())
        
        for key in all_keys:
            old_val = old_data.get(key)
            new_val = new_data.get(key)
            
            if old_val != new_val:
                diff[key] = {
                    "old": old_val,
                    "new": new_val
                }
        
        return diff

# Custom State Types
class WorkflowState(TypedDict):
    """Custom workflow state with enhanced structure"""
    state_id: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_step: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: float
    metadata: Dict[str, Any]
    context: Dict[str, Any]
    history: List[Dict[str, Any]]

class SessionState(TypedDict):
    """Session-specific state"""
    session_id: str
    user_id: str
    session_data: Dict[str, Any]
    preferences: Dict[str, Any]
    last_activity: float
    session_metadata: Dict[str, Any]

# Global state manager
state_manager = StateManager()

# Workflow nodes using custom state management
def initialize_workflow_state(state: WorkflowState) -> WorkflowState:
    """Initialize workflow with custom state management"""
    state_id = state.get("state_id", f"workflow_{int(time.time())}")
    
    # Create state in manager
    initial_data = {
        "state_id": state_id,
        "messages": list(state.get("messages", [])),
        "current_step": "initialization",
        "status": "pending",
        "progress": 0.0,
        "metadata": state.get("metadata", {}),
        "context": state.get("context", {}),
        "history": []
    }
    
    state_manager.create_state(
        state_id=state_id,
        initial_data=initial_data,
        state_type=StateType.WORKFLOW,
        priority=StatePriority.HIGH,
        tags=["workflow", "initialization"]
    )
    
    # Create initial snapshot
    state_manager.create_snapshot(state_id, "initial")
    
    print(f"🚀 Initialized workflow state: {state_id}")
    
    return state

def process_with_state_management(state: WorkflowState) -> WorkflowState:
    """Process step with custom state management"""
    state_id = state["state_id"]
    
    # Read current state
    current_state = state_manager.read_state(state_id)
    if not current_state:
        print(f"❌ Could not read state {state_id}")
        return state
    
    # Update state
    update_data = {
        "current_step": "processing",
        "status": "running",
        "progress": 0.5,
        "context": {
            **current_state.get("context", {}),
            "processing_started": time.time()
        }
    }
    
    state_manager.update_state(state_id, update_data)
    
    # Add to history
    history_entry = {
        "step": "processing",
        "timestamp": time.time(),
        "action": "Started processing"
    }
    
    history_update = {
        "history": current_state.get("history", []) + [history_entry]
    }
    
    state_manager.update_state(state_id, history_update)
    
    print(f"⚙️ Processing step completed for state: {state_id}")
    
    return state

def finalize_with_state_management(state: WorkflowState) -> WorkflowState:
    """Finalize workflow with custom state management"""
    state_id = state["state_id"]
    
    # Read current state
    current_state = state_manager.read_state(state_id)
    if not current_state:
        print(f"❌ Could not read state {state_id}")
        return state
    
    # Create final snapshot before completion
    state_manager.create_snapshot(state_id, "pre-finalization")
    
    # Update state to completed
    update_data = {
        "current_step": "completed",
        "status": "completed",
        "progress": 1.0,
        "context": {
            **current_state.get("context", {}),
            "completed_at": time.time()
        }
    }
    
    state_manager.update_state(state_id, update_data)
    
    # Create final snapshot
    state_manager.create_snapshot(state_id, "final")
    
    print(f"✅ Finalized workflow state: {state_id}")
    
    return state

def create_custom_state_management_graph():
    """Create graph with custom state management"""
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_workflow_state)
    workflow.add_node("process", process_with_state_management)
    workflow.add_node("finalize", finalize_with_state_management)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "process")
    workflow.add_edge("process", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

def demonstrate_custom_state_management():
    """Demonstrate custom state management features"""
    print("\n" + "="*80)
    print("CUSTOM STATE MANAGEMENT DEMONSTRATION")
    print("="*80)
    
    # Create graph
    graph = create_custom_state_management_graph()
    
    # Set up initial state
    initial_state = {
        "state_id": "demo_workflow_1",
        "messages": [HumanMessage(content="Start custom state management demo")],
        "current_step": "",
        "status": "pending",
        "progress": 0.0,
        "metadata": {"demo": True},
        "context": {},
        "history": []
    }
    
    config = {"configurable": {"thread_id": "custom-state-demo"}}
    
    print("\n🚀 Running workflow with custom state management and visualization...")
    
    # Use the context manager to run with visualization
    with visualize(graph) as viz_app:
        print("Running with visualization - Browser will open at http://localhost:8765")
        
        # IMPORTANT: Use viz_app, not graph
        result = viz_app.invoke(initial_state, config=config)
        
        print(f"\n📊 Workflow completed:")
        print(f"- State ID: {result['state_id']}")
        print(f"- Final Status: {result['status']}")
        print(f"- Final Progress: {result['progress']}")
        
        # Demonstrate state manager features
        print(f"\n🔍 State Manager Analysis:")
        
        # List all states
        all_states = state_manager.list_states()
        print(f"- Total states: {len(all_states)}")
        print(f"- State IDs: {all_states}")
        
        # Get detailed state info
        if all_states:
            state_info = state_manager.get_state_info(all_states[0])
            if state_info:
                print(f"\n📋 State Details for {all_states[0]}:")
                print(f"- Version: {state_info['metadata']['version']}")
                print(f"- Size: {state_info['metadata']['size_bytes']} bytes")
                print(f"- Access Count: {state_info['metadata']['access_count']}")
                print(f"- Snapshots: {state_info['snapshot_count']}")
        
        # List snapshots
        print(f"\n📸 Available Snapshots:")
        for snapshot_id, snapshot in state_manager.snapshots.items():
            print(f"- {snapshot_id}: {snapshot.timestamp} (version {snapshot.metadata.version})")
        
        # Demonstrate state restoration
        if state_manager.snapshots:
            first_snapshot_id = list(state_manager.snapshots.keys())[0]
            print(f"\n🔄 Restoring from snapshot: {first_snapshot_id}")
            
            restored_state_id = f"restored_{int(time.time())}"
            success = state_manager.restore_snapshot(first_snapshot_id, restored_state_id)
            
            if success:
                restored_info = state_manager.get_state_info(restored_state_id)
                print(f"✅ Restored state: {restored_state_id}")
                print(f"- Restored version: {restored_info['metadata']['version']}")

def demonstrate_state_validation_and_transformation():
    """Demonstrate state validation and transformation"""
    print("\n" + "="*80)
    print("STATE VALIDATION AND TRANSFORMATION DEMONSTRATION")
    print("="*80)
    
    # Test state validation
    print("\n🔍 Testing State Validation:")
    
    valid_state = {
        "messages": [HumanMessage(content="Valid message")],
        "current_step": "test",
        "status": "running"
    }
    
    invalid_state = {
        "messages": "not a list",
        "current_step": "test",
        "status": "invalid_status"
    }
    
    validator = WorkflowStateValidator()
    
    print(f"- Valid state validation: {validator.validate(valid_state)}")
    print(f"- Invalid state validation: {validator.validate(invalid_state)}")
    
    if not validator.validate(invalid_state):
        errors = validator.get_validation_errors(invalid_state)
        print(f"- Validation errors: {errors}")
    
    # Test state transformation
    print(f"\n🔄 Testing State Transformation:")
    
    # Test message normalizer
    normalizer = MessageNormalizer()
    test_state = {
        "messages": [
            {"type": "human", "content": "Hello"},
            {"type": "ai", "content": "Hi there!"}
        ]
    }
    
    normalized_state = normalizer.transform(test_state)
    print(f"- Original messages: {len(test_state['messages'])} dict messages")
    print(f"- Normalized messages: {len(normalized_state['messages'])} BaseMessage objects")
    
    # Test state compression
    compressor = StateCompressor()
    large_state = {
        "large_list": list(range(200)),
        "large_string": "x" * 1500,
        "normal_data": "small"
    }
    
    compressed_state = compressor.transform(large_state)
    print(f"- Original list size: {len(large_state['large_list'])}")
    print(f"- Compressed list size: {len(compressed_state['large_list'])}")
    print(f"- Original string length: {len(large_state['large_string'])}")
    print(f"- Compressed string length: {len(compressed_state['large_string'])}")

# Main execution
if __name__ == "__main__":
    # Create the graph
    custom_state_graph = create_custom_state_management_graph()
    
    # Display the graph structure
    try:
        display(Image(custom_state_graph.get_graph().draw_mermaid_png()))
    except:
        print("Graph visualization not available")
    
    print("\n" + "="*80)
    print("CUSTOM STATE MANAGEMENT DEMONSTRATIONS")
    print("="*80)
    
    print("\n🚀 Starting demonstrations with visualization...")
    
    # Use the context manager to run with visualization
    with visualize(custom_state_graph) as viz_app:
        print("Running with visualization - Browser will open at http://localhost:8765")
        
        # Demonstrate custom state management
        print("\n--- CUSTOM STATE MANAGEMENT DEMONSTRATION ---")
        
        # Set up initial state
        initial_state = {
            "state_id": "demo_workflow_1",
            "messages": [HumanMessage(content="Start custom state management demo")],
            "current_step": "",
            "status": "pending",
            "progress": 0.0,
            "metadata": {"demo": True},
            "context": {},
            "history": []
        }
        
        config = {"configurable": {"thread_id": "custom-state-demo"}}
        
        print("🚀 Running workflow with custom state management...")
        
        # IMPORTANT: Use viz_app, not custom_state_graph
        result = viz_app.invoke(initial_state, config=config)
        
        print(f"\n📊 Workflow completed:")
        print(f"- State ID: {result['state_id']}")
        print(f"- Final Status: {result['status']}")
        print(f"- Final Progress: {result['progress']}")
        
        # Demonstrate state manager features
        print(f"\n🔍 State Manager Analysis:")
        
        # List all states
        all_states = state_manager.list_states()
        print(f"- Total states: {len(all_states)}")
        print(f"- State IDs: {all_states}")
        
        # Get detailed state info
        if all_states:
            state_info = state_manager.get_state_info(all_states[0])
            if state_info:
                print(f"\n📋 State Details for {all_states[0]}:")
                print(f"- Version: {state_info['metadata']['version']}")
                print(f"- Size: {state_info['metadata']['size_bytes']} bytes")
                print(f"- Access Count: {state_info['metadata']['access_count']}")
                print(f"- Snapshots: {state_info['snapshot_count']}")
        
        # List snapshots
        print(f"\n📸 Available Snapshots:")
        for snapshot_id, snapshot in state_manager.snapshots.items():
            print(f"- {snapshot_id}: {snapshot.timestamp} (version {snapshot.metadata.version})")
        
        # Demonstrate state restoration
        if state_manager.snapshots:
            first_snapshot_id = list(state_manager.snapshots.keys())[0]
            print(f"\n🔄 Restoring from snapshot: {first_snapshot_id}")
            
            restored_state_id = f"restored_{int(time.time())}"
            success = state_manager.restore_snapshot(first_snapshot_id, restored_state_id)
            
            if success:
                restored_info = state_manager.get_state_info(restored_state_id)
                print(f"✅ Restored state: {restored_state_id}")
                print(f"- Restored version: {restored_info['metadata']['version']}")
        
        # Demonstrate validation and transformation
        print("\n--- STATE VALIDATION AND TRANSFORMATION DEMONSTRATION ---")
        
        # Test state validation
        print(f"\n🔍 Testing State Validation:")
        
        valid_state = {
            "messages": [HumanMessage(content="Valid message")],
            "current_step": "test",
            "status": "running"
        }
        
        invalid_state = {
            "messages": "not a list",
            "current_step": "test",
            "status": "invalid_status"
        }
        
        validator = WorkflowStateValidator()
        
        print(f"- Valid state validation: {validator.validate(valid_state)}")
        print(f"- Invalid state validation: {validator.validate(invalid_state)}")
        
        if not validator.validate(invalid_state):
            errors = validator.get_validation_errors(invalid_state)
            print(f"- Validation errors: {errors}")
        
        # Test state transformation
        print(f"\n🔄 Testing State Transformation:")
        
        # Test message normalizer
        normalizer = MessageNormalizer()
        test_state = {
            "messages": [
                {"type": "human", "content": "Hello"},
                {"type": "ai", "content": "Hi there!"}
            ]
        }
        
        normalized_state = normalizer.transform(test_state)
        print(f"- Original messages: {len(test_state['messages'])} dict messages")
        print(f"- Normalized messages: {len(normalized_state['messages'])} BaseMessage objects")
        
        # Test state compression
        compressor = StateCompressor()
        large_state = {
            "large_list": list(range(200)),
            "large_string": "x" * 1500,
            "normal_data": "small"
        }
        
        compressed_state = compressor.transform(large_state)
        print(f"- Original list size: {len(large_state['large_list'])}")
        print(f"- Compressed list size: {len(compressed_state['large_list'])}")
        print(f"- Original string length: {len(large_state['large_string'])}")
        print(f"- Compressed string length: {len(compressed_state['large_string'])}")
    
    print("\n" + "="*80)
    print("ALL DEMONSTRATIONS COMPLETED - Visualization server closed")
    print("="*80)
    
    # Final summary
    print(f"\n📊 Final State Manager Summary:")
    print(f"- Total states managed: {len(state_manager.states)}")
    print(f"- Total snapshots created: {len(state_manager.snapshots)}")
    print(f"- State types in use: {list(set(meta.state_type.value for meta in state_manager.metadata.values()))}")
    print(f"- Total state size: {sum(meta.size_bytes for meta in state_manager.metadata.values())} bytes")
