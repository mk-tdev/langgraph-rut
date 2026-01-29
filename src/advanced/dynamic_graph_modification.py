"""
Dynamic Graph Modification in LangGraph

This example demonstrates advanced dynamic graph modification patterns including:
- Runtime node addition and removal
- Dynamic edge creation and deletion
- Conditional graph structure changes
- Adaptive workflow evolution
- Runtime graph reconfiguration
"""

import json
import time
from typing import Literal, TypedDict, Annotated, Sequence, List, Dict, Any, Optional, Callable
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from pydantic import BaseModel, Field
import uuid
from langgraph_viz import visualize

# Initialize LLM
llm = ChatOllama(model="gpt-oss:120b-cloud")

class ModificationType(str, Enum):
    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    MODIFY_NODE = "modify_node"
    RECONFIGURE_GRAPH = "reconfigure_graph"

class NodeType(str, Enum):
    PROCESSOR = "processor"
    VALIDATOR = "validator"
    TRANSFORMER = "transformer"
    AGGREGATOR = "aggregator"
    FILTER = "filter"
    CUSTOM = "custom"

class GraphModification(BaseModel):
    """Model for graph modification operations"""
    modification_id: str = Field(description="Unique identifier for the modification")
    modification_type: ModificationType = Field(description="Type of modification")
    target_node: Optional[str] = Field(description="Target node for modification")
    source_node: Optional[str] = Field(description="Source node for edge modifications")
    destination_node: Optional[str] = Field(description="Destination node for edge modifications")
    node_function: Optional[Callable] = Field(description="Function to add for new nodes")
    modification_data: Dict[str, Any] = Field(description="Additional modification data")
    timestamp: float = Field(description="Timestamp of the modification")
    reason: str = Field(description="Reason for the modification")

class DynamicGraphState(TypedDict):
    """State for dynamic graph modification workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_graph: Dict[str, Any]
    active_nodes: List[str]
    active_edges: List[Dict[str, str]]
    modification_history: List[GraphModification]
    graph_metadata: Dict[str, Any]
    adaptation_triggers: List[str]
    performance_metrics: Dict[str, float]

class DynamicGraphManager:
    """Manages dynamic graph modifications"""
    
    def __init__(self):
        self.modifications = []
        self.node_registry = {}
        self.edge_registry = []
        self.graph_snapshots = []
    
    def register_node(self, node_name: str, node_function: Callable, node_type: NodeType):
        """Register a node that can be added dynamically"""
        self.node_registry[node_name] = {
            "function": node_function,
            "type": node_type,
            "registered_at": time.time()
        }
    
    def add_node_to_graph(self, workflow: StateGraph, node_name: str, node_function: Callable):
        """Add a node to the workflow"""
        workflow.add_node(node_name, node_function)
        print(f"➕ Added node: {node_name}")
    
    def remove_node_from_graph(self, workflow: StateGraph, node_name: str):
        """Remove a node from the workflow (conceptual - actual removal requires graph rebuild)"""
        # In practice, this would require rebuilding the graph
        print(f"➖ Marked node for removal: {node_name}")
    
    def add_edge_to_graph(self, workflow: StateGraph, source: str, destination: str):
        """Add an edge to the workflow"""
        workflow.add_edge(source, destination)
        print(f"🔗 Added edge: {source} -> {destination}")
    
    def remove_edge_from_graph(self, workflow: StateGraph, source: str, destination: str):
        """Remove an edge from the workflow (conceptual)"""
        print(f"✂️ Marked edge for removal: {source} -> {destination}")
    
    def create_graph_snapshot(self, workflow: StateGraph, snapshot_name: str):
        """Create a snapshot of the current graph structure"""
        snapshot = {
            "name": snapshot_name,
            "timestamp": time.time(),
            "nodes": list(workflow.nodes.keys()) if hasattr(workflow, 'nodes') else [],
            "edges": []  # Would need to extract from graph structure
        }
        self.graph_snapshots.append(snapshot)
        print(f"📸 Created graph snapshot: {snapshot_name}")
    
    def list_modifications(self) -> List[GraphModification]:
        """List all modifications made"""
        return self.modifications

# Global graph manager
graph_manager = DynamicGraphManager()

# Dynamic node functions
def data_validator(state: DynamicGraphState) -> DynamicGraphState:
    """Dynamic data validation node"""
    print("🔍 Running dynamic data validation")
    
    validation_result = {
        "valid": True,
        "issues": [],
        "validation_time": time.time()
    }
    
    state["graph_metadata"]["validation_results"] = validation_result
    state["messages"].append(AIMessage(content="✅ Dynamic validation completed"))
    
    return state

def data_enricher(state: DynamicGraphState) -> DynamicGraphState:
    """Dynamic data enrichment node"""
    print("🎨 Running dynamic data enrichment")
    
    enrichment_result = {
        "enriched_fields": ["category", "priority", "source"],
        "enrichment_time": time.time()
    }
    
    state["graph_metadata"]["enrichment_results"] = enrichment_result
    state["messages"].append(AIMessage(content="✨ Dynamic enrichment completed"))
    
    return state

def performance_monitor(state: DynamicGraphState) -> DynamicGraphState:
    """Dynamic performance monitoring node"""
    print("📊 Running dynamic performance monitoring")
    
    performance_data = {
        "processing_time": time.time() - state["graph_metadata"].get("start_time", time.time()),
        "node_count": len(state["active_nodes"]),
        "edge_count": len(state["active_edges"]),
        "memory_usage": 0.75  # Simulated
    }
    
    state["performance_metrics"] = performance_data
    state["messages"].append(AIMessage(content="📈 Performance monitoring completed"))
    
    return state

def adaptive_router(state: DynamicGraphState) -> DynamicGraphState:
    """Adaptive routing that can trigger graph modifications"""
    print("🧭 Running adaptive routing analysis")
    
    # Analyze current performance and state
    performance_metrics = state.get("performance_metrics", {})
    active_nodes = state.get("active_nodes", [])
    
    # Determine if modifications are needed
    modifications_needed = []
    
    # Check if performance is poor
    if performance_metrics.get("memory_usage", 0) > 0.8:
        modifications_needed.append("add_optimization_node")
    
    # Check if validation is missing
    if not any("validator" in node for node in active_nodes):
        modifications_needed.append("add_validator_node")
    
    # Check if enrichment is missing
    if not any("enricher" in node for node in active_nodes):
        modifications_needed.append("add_enricher_node")
    
    state["adaptation_triggers"] = modifications_needed
    
    if modifications_needed:
        trigger_message = f"🔄 Adaptation triggers detected: {', '.join(modifications_needed)}"
        state["messages"].append(AIMessage(content=trigger_message))
    else:
        state["messages"].append(AIMessage(content="✅ No adaptations needed"))
    
    return state

def graph_modifier(state: DynamicGraphState) -> DynamicGraphState:
    """Dynamic graph modification node"""
    print("🔧 Running dynamic graph modification")
    
    adaptation_triggers = state.get("adaptation_triggers", [])
    
    for trigger in adaptation_triggers:
        if trigger == "add_validator_node":
            modification = GraphModification(
                modification_id=str(uuid.uuid4()),
                modification_type=ModificationType.ADD_NODE,
                target_node="dynamic_validator",
                node_function=data_validator,
                modification_data={"node_type": NodeType.VALIDATOR},
                timestamp=time.time(),
                reason="Performance optimization: adding validation"
            )
            state["modification_history"].append(modification)
            state["active_nodes"].append("dynamic_validator")
            
        elif trigger == "add_enricher_node":
            modification = GraphModification(
                modification_id=str(uuid.uuid4()),
                modification_type=ModificationType.ADD_NODE,
                target_node="dynamic_enricher",
                node_function=data_enricher,
                modification_data={"node_type": NodeType.TRANSFORMER},
                timestamp=time.time(),
                reason="Feature enhancement: adding data enrichment"
            )
            state["modification_history"].append(modification)
            state["active_nodes"].append("dynamic_enricher")
            
        elif trigger == "add_optimization_node":
            modification = GraphModification(
                modification_id=str(uuid.uuid4()),
                modification_type=ModificationType.ADD_NODE,
                target_node="performance_monitor",
                node_function=performance_monitor,
                modification_data={"node_type": NodeType.AGGREGATOR},
                timestamp=time.time(),
                reason="Performance monitoring: adding optimization"
            )
            state["modification_history"].append(modification)
            state["active_nodes"].append("performance_monitor")
    
    modification_count = len(state["modification_history"])
    state["messages"].append(AIMessage(content=f"🔧 Applied {modification_count} graph modifications"))
    
    return state

def dynamic_processor(state: DynamicGraphState) -> DynamicGraphState:
    """Dynamic data processing node"""
    print("⚙️ Running dynamic data processing")
    
    processing_result = {
        "processed_items": 100,
        "processing_time": time.time(),
        "active_nodes": len(state["active_nodes"])
    }
    
    state["graph_metadata"]["processing_results"] = processing_result
    state["messages"].append(AIMessage(content="⚙️ Dynamic processing completed"))
    
    return state

def create_initial_graph():
    """Create the initial dynamic graph"""
    
    workflow = StateGraph(DynamicGraphState)
    
    # Register dynamic nodes
    graph_manager.register_node("dynamic_validator", data_validator, NodeType.VALIDATOR)
    graph_manager.register_node("dynamic_enricher", data_enricher, NodeType.TRANSFORMER)
    graph_manager.register_node("performance_monitor", performance_monitor, NodeType.AGGREGATOR)
    
    # Add initial nodes
    workflow.add_node("dynamic_processor", dynamic_processor)
    workflow.add_node("adaptive_router", adaptive_router)
    workflow.add_node("graph_modifier", graph_modifier)
    
    # Add initial edges
    workflow.add_edge(START, "dynamic_processor")
    workflow.add_edge("dynamic_processor", "adaptive_router")
    workflow.add_edge("adaptive_router", "graph_modifier")
    workflow.add_edge("graph_modifier", END)
    
    return workflow.compile()

def create_modified_graph(modifications: List[GraphModification]):
    """Create a modified graph based on modification history"""
    
    workflow = StateGraph(DynamicGraphState)
    
    # Add base nodes
    workflow.add_node("dynamic_processor", dynamic_processor)
    workflow.add_node("adaptive_router", adaptive_router)
    workflow.add_node("graph_modifier", graph_modifier)
    
    # Apply modifications
    active_nodes = ["dynamic_processor", "adaptive_router", "graph_modifier"]
    
    for modification in modifications:
        if modification.modification_type == ModificationType.ADD_NODE:
            if modification.node_function:
                workflow.add_node(modification.target_node, modification.node_function)
                active_nodes.append(modification.target_node)
                print(f"🔧 Applied modification: Added node {modification.target_node}")
    
    # Rebuild edges based on active nodes
    workflow.add_edge(START, "dynamic_processor")
    workflow.add_edge("dynamic_processor", "adaptive_router")
    
    # Add edges to dynamically added nodes
    if "dynamic_validator" in active_nodes:
        workflow.add_edge("adaptive_router", "dynamic_validator")
        workflow.add_edge("dynamic_validator", "graph_modifier")
    
    if "dynamic_enricher" in active_nodes:
        workflow.add_edge("adaptive_router", "dynamic_enricher")
        workflow.add_edge("dynamic_enricher", "graph_modifier")
    
    if "performance_monitor" in active_nodes:
        workflow.add_edge("graph_modifier", "performance_monitor")
        workflow.add_edge("performance_monitor", END)
    else:
        workflow.add_edge("graph_modifier", END)
    
    return workflow.compile(), active_nodes

def demonstrate_dynamic_modification():
    """Demonstrate dynamic graph modification"""
    print("\n" + "="*80)
    print("DYNAMIC GRAPH MODIFICATION DEMONSTRATION")
    print("="*80)
    
    # Create initial graph
    initial_graph = create_initial_graph()
    
    # Set up initial state
    initial_state = {
        "messages": [HumanMessage(content="Start dynamic graph modification demo")],
        "current_graph": {"version": "1.0", "nodes": 3, "edges": 3},
        "active_nodes": ["dynamic_processor", "adaptive_router", "graph_modifier"],
        "active_edges": [
            {"source": "START", "target": "dynamic_processor"},
            {"source": "dynamic_processor", "target": "adaptive_router"},
            {"source": "adaptive_router", "target": "graph_modifier"},
            {"source": "graph_modifier", "target": "END"}
        ],
        "modification_history": [],
        "graph_metadata": {"start_time": time.time()},
        "adaptation_triggers": [],
        "performance_metrics": {}
    }
    
    config = {"configurable": {"thread_id": "dynamic-modification-demo"}}
    
    print("\n🚀 Running initial graph with visualization...")
    
    # Use the context manager to run with visualization
    with visualize(initial_graph) as viz_app:
        print("Running with visualization - Browser will open at http://localhost:8765")
        
        # IMPORTANT: Use viz_app, not initial_graph
        result1 = viz_app.invoke(initial_state, config=config)
        
        print(f"\n📊 Initial execution completed:")
        print(f"- Active nodes: {result1['active_nodes']}")
        print(f"- Modification history: {len(result1['modification_history'])} modifications")
        print(f"- Adaptation triggers: {result1['adaptation_triggers']}")
        
        # Display modifications
        if result1["modification_history"]:
            print(f"\n🔧 Modifications applied:")
            for mod in result1["modification_history"]:
                print(f"  - {mod.modification_type.value}: {mod.target_node} ({mod.reason})")
        
        # Create modified graph
        print(f"\n🔄 Creating modified graph based on adaptations...")
        
        modified_graph, new_active_nodes = create_modified_graph(result1["modification_history"])
        
        # Update state with new graph structure
        modified_state = result1.copy()
        modified_state["active_nodes"] = new_active_nodes
        modified_state["current_graph"] = {
            "version": "2.0",
            "nodes": len(new_active_nodes),
            "edges": len(new_active_nodes) + 1  # Approximate
        }
        
        print(f"\n🚀 Running modified graph...")
        
        # IMPORTANT: Use viz_app, not modified_graph
        result2 = viz_app.invoke(modified_state, config=config)
        
        print(f"\n📊 Modified execution completed:")
        print(f"- Active nodes: {result2['active_nodes']}")
        print(f"- Total modifications: {len(result2['modification_history'])}")
        print(f"- Performance metrics: {result2['performance_metrics']}")

def demonstrate_adaptive_evolution():
    """Demonstrate adaptive graph evolution over multiple iterations"""
    print("\n" + "="*80)
    print("ADAPTIVE GRAPH EVOLUTION DEMONSTRATION")
    print("="*80)
    
    # Start with basic graph
    current_state = {
        "messages": [HumanMessage(content="Start adaptive evolution")],
        "current_graph": {"version": "1.0", "nodes": 3, "edges": 3},
        "active_nodes": ["dynamic_processor", "adaptive_router", "graph_modifier"],
        "active_edges": [
            {"source": "START", "target": "dynamic_processor"},
            {"source": "dynamic_processor", "target": "adaptive_router"},
            {"source": "adaptive_router", "target": "graph_modifier"},
            {"source": "graph_modifier", "target": "END"}
        ],
        "modification_history": [],
        "graph_metadata": {"start_time": time.time()},
        "adaptation_triggers": [],
        "performance_metrics": {}
    }
    
    config = {"configurable": {"thread_id": "adaptive-evolution-demo"}}
    
    # Run multiple iterations to show evolution
    for iteration in range(3):
        print(f"\n--- Evolution Iteration {iteration + 1} ---")
        
        # Create graph for current iteration
        if iteration == 0:
            current_graph = create_initial_graph()
        else:
            current_graph, current_state["active_nodes"] = create_modified_graph(current_state["modification_history"])
        
        # Simulate changing conditions
        if iteration == 1:
            # Simulate high memory usage
            current_state["performance_metrics"] = {"memory_usage": 0.85}
        elif iteration == 2:
            # Simulate need for enrichment
            current_state["adaptation_triggers"] = ["add_enricher_node"]
        
        # Run current iteration
        result = current_graph.invoke(current_state, config=config)
        
        print(f"Graph version: {result['current_graph']['version']}")
        print(f"Active nodes: {result['active_nodes']}")
        print(f"New modifications: {len(result['modification_history'])}")
        
        # Update state for next iteration
        current_state = result
    
    print(f"\n🎉 Evolution completed!")
    print(f"Final graph version: {current_state['current_graph']['version']}")
    print(f"Final active nodes: {current_state['active_nodes']}")
    print(f"Total modifications: {len(current_state['modification_history'])}")

def demonstrate_runtime_reconfiguration():
    """Demonstrate runtime graph reconfiguration"""
    print("\n" + "="*80)
    print("RUNTIME RECONFIGURATION DEMONSTRATION")
    print("="*80)
    
    # Create initial graph
    graph = create_initial_graph()
    
    # Set up state
    state = {
        "messages": [HumanMessage(content="Start runtime reconfiguration")],
        "current_graph": {"version": "1.0", "nodes": 3, "edges": 3},
        "active_nodes": ["dynamic_processor", "adaptive_router", "graph_modifier"],
        "active_edges": [
            {"source": "START", "target": "dynamic_processor"},
            {"source": "dynamic_processor", "target": "adaptive_router"},
            {"source": "adaptive_router", "target": "graph_modifier"},
            {"source": "graph_modifier", "target": "END"}
        ],
        "modification_history": [],
        "graph_metadata": {"start_time": time.time()},
        "adaptation_triggers": [],
        "performance_metrics": {}
    }
    
    config = {"configurable": {"thread_id": "runtime-reconfig-demo"}}
    
    print("\n🚀 Initial graph execution...")
    
    # Run initial execution
    result = graph.invoke(state, config=config)
    
    print(f"Initial result: {len(result['active_nodes'])} nodes")
    
    # Simulate runtime condition change
    print("\n🔄 Runtime condition detected: High memory usage")
    
    # Create new modification
    memory_optimization_mod = GraphModification(
        modification_id=str(uuid.uuid4()),
        modification_type=ModificationType.ADD_NODE,
        target_node="performance_monitor",
        node_function=performance_monitor,
        modification_data={"node_type": NodeType.AGGREGATOR},
        timestamp=time.time(),
        reason="Runtime optimization: memory usage too high"
    )
    
    result["modification_history"].append(memory_optimization_mod)
    
    # Reconfigure graph
    print("\n🔧 Reconfiguring graph at runtime...")
    
    reconfigured_graph, new_active_nodes = create_modified_graph(result["modification_history"])
    
    # Update state
    result["active_nodes"] = new_active_nodes
    result["current_graph"]["version"] = "1.1"
    result["current_graph"]["nodes"] = len(new_active_nodes)
    
    print(f"✅ Graph reconfigured: {len(new_active_nodes)} active nodes")
    
    # Run reconfigured graph
    print("\n🚀 Running reconfigured graph...")
    
    final_result = reconfigured_graph.invoke(result, config=config)
    
    print(f"✅ Reconfigured execution completed")
    print(f"Final nodes: {final_result['active_nodes']}")
    print(f"Performance metrics: {final_result['performance_metrics']}")

# Main execution
if __name__ == "__main__":
    # Create the initial graph
    dynamic_graph = create_initial_graph()
    
    # Display the graph structure
    try:
        display(Image(dynamic_graph.get_graph().draw_mermaid_png()))
    except:
        print("Graph visualization not available")
    
    print("\n" + "="*80)
    print("DYNAMIC GRAPH MODIFICATION DEMONSTRATIONS")
    print("="*80)
    
    print("\n🚀 Starting demonstrations with visualization...")
    
    # Use the context manager to run with visualization
    with visualize(dynamic_graph) as viz_app:
        print("Running with visualization - Browser will open at http://localhost:8765")
        
        # Demonstrate basic dynamic modification
        print("\n--- DYNAMIC MODIFICATION DEMONSTRATION ---")
        
        # Set up initial state
        initial_state = {
            "messages": [HumanMessage(content="Start dynamic graph modification demo")],
            "current_graph": {"version": "1.0", "nodes": 3, "edges": 3},
            "active_nodes": ["dynamic_processor", "adaptive_router", "graph_modifier"],
            "active_edges": [
                {"source": "START", "target": "dynamic_processor"},
                {"source": "dynamic_processor", "target": "adaptive_router"},
                {"source": "adaptive_router", "target": "graph_modifier"},
                {"source": "graph_modifier", "target": "END"}
            ],
            "modification_history": [],
            "graph_metadata": {"start_time": time.time()},
            "adaptation_triggers": [],
            "performance_metrics": {}
        }
        
        config = {"configurable": {"thread_id": "dynamic-modification-demo"}}
        
        print("🚀 Running initial graph...")
        
        # IMPORTANT: Use viz_app, not dynamic_graph
        result1 = viz_app.invoke(initial_state, config=config)
        
        print(f"\n📊 Initial execution completed:")
        print(f"- Active nodes: {result1['active_nodes']}")
        print(f"- Modification history: {len(result1['modification_history'])} modifications")
        print(f"- Adaptation triggers: {result1['adaptation_triggers']}")
        
        # Demonstrate adaptive evolution
        print("\n--- ADAPTIVE EVOLUTION DEMONSTRATION ---")
        
        # Start with basic state for evolution
        evolution_state = {
            "messages": [HumanMessage(content="Start adaptive evolution")],
            "current_graph": {"version": "1.0", "nodes": 3, "edges": 3},
            "active_nodes": ["dynamic_processor", "adaptive_router", "graph_modifier"],
            "active_edges": [
                {"source": "START", "target": "dynamic_processor"},
                {"source": "dynamic_processor", "target": "adaptive_router"},
                {"source": "adaptive_router", "target": "graph_modifier"},
                {"source": "graph_modifier", "target": "END"}
            ],
            "modification_history": [],
            "graph_metadata": {"start_time": time.time()},
            "adaptation_triggers": [],
            "performance_metrics": {}
        }
        
        evolution_config = {"configurable": {"thread_id": "adaptive-evolution-demo"}}
        
        # Run multiple iterations to show evolution
        for iteration in range(3):
            print(f"\n--- Evolution Iteration {iteration + 1} ---")
            
            # Simulate changing conditions
            if iteration == 1:
                # Simulate high memory usage
                evolution_state["performance_metrics"] = {"memory_usage": 0.85}
            elif iteration == 2:
                # Simulate need for enrichment
                evolution_state["adaptation_triggers"] = ["add_enricher_node"]
            
            print(f"Graph version: {evolution_state['current_graph']['version']}")
            print(f"Active nodes: {evolution_state['active_nodes']}")
            print(f"New modifications: {len(evolution_state['modification_history'])}")
            
            # IMPORTANT: Use viz_app, not dynamic_graph
            evolution_result = viz_app.invoke(evolution_state, config=evolution_config)
            
            print(f"Evolution {iteration + 1} completed")
            
            # Update state for next iteration
            evolution_state = evolution_result
        
        print(f"\n🎉 Evolution completed!")
        print(f"Final graph version: {evolution_state['current_graph']['version']}")
        print(f"Final active nodes: {evolution_state['active_nodes']}")
        print(f"Total modifications: {len(evolution_state['modification_history'])}")
        
        # Demonstrate runtime reconfiguration
        print("\n--- RUNTIME RECONFIGURATION DEMONSTRATION ---")
        
        reconfig_state = {
            "messages": [HumanMessage(content="Start runtime reconfiguration")],
            "current_graph": {"version": "1.0", "nodes": 3, "edges": 3},
            "active_nodes": ["dynamic_processor", "adaptive_router", "graph_modifier"],
            "active_edges": [
                {"source": "START", "target": "dynamic_processor"},
                {"source": "dynamic_processor", "target": "adaptive_router"},
                {"source": "adaptive_router", "target": "graph_modifier"},
                {"source": "graph_modifier", "target": "END"}
            ],
            "modification_history": [],
            "graph_metadata": {"start_time": time.time()},
            "adaptation_triggers": [],
            "performance_metrics": {}
        }
        
        reconfig_config = {"configurable": {"thread_id": "runtime-reconfig-demo"}}
        
        print("🚀 Initial graph execution...")
        
        # IMPORTANT: Use viz_app, not dynamic_graph
        reconfig_result = viz_app.invoke(reconfig_state, config=reconfig_config)
        
        print(f"Initial result: {len(reconfig_result['active_nodes'])} nodes")
        
        # Simulate runtime condition change
        print("\n🔄 Runtime condition detected: High memory usage")
        
        # Create new modification
        memory_optimization_mod = GraphModification(
            modification_id=str(uuid.uuid4()),
            modification_type=ModificationType.ADD_NODE,
            target_node="performance_monitor",
            node_function=performance_monitor,
            modification_data={"node_type": NodeType.AGGREGATOR},
            timestamp=time.time(),
            reason="Runtime optimization: memory usage too high"
        )
        
        reconfig_result["modification_history"].append(memory_optimization_mod)
        
        # Reconfigure graph
        print("\n🔧 Reconfiguring graph at runtime...")
        
        reconfigured_graph, new_active_nodes = create_modified_graph(reconfig_result["modification_history"])
        
        # Update state
        reconfig_result["active_nodes"] = new_active_nodes
        reconfig_result["current_graph"]["version"] = "1.1"
        reconfig_result["current_graph"]["nodes"] = len(new_active_nodes)
        
        print(f"✅ Graph reconfigured: {len(new_active_nodes)} active nodes")
        
        # Run reconfigured graph
        print("\n🚀 Running reconfigured graph...")
        
        # IMPORTANT: Use viz_app, not reconfigured_graph
        final_reconfig_result = viz_app.invoke(reconfig_result, config=reconfig_config)
        
        print(f"✅ Reconfigured execution completed")
        print(f"Final nodes: {final_reconfig_result['active_nodes']}")
        print(f"Performance metrics: {final_reconfig_result['performance_metrics']}")
    
    print("\n" + "="*80)
    print("ALL DEMONSTRATIONS COMPLETED - Visualization server closed")
    print("="*80)
    
    # Final summary
    all_modifications = graph_manager.list_modifications()
    print(f"\n📊 Final summary:")
    print(f"- Total modifications registered: {len(all_modifications)}")
    print(f"- Node registry size: {len(graph_manager.node_registry)}")
    print(f"- Graph snapshots: {len(graph_manager.graph_snapshots)}")
