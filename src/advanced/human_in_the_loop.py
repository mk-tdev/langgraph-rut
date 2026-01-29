"""
Human-in-the-Loop Workflows in LangGraph

This example demonstrates advanced human-in-the-loop patterns including:
- Approval workflows with human confirmation
- Interactive decision points
- Manual review and correction processes
- Feedback incorporation loops
- Breakpoints for human intervention
"""

import json
from typing import Literal, TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from pydantic import BaseModel, Field
import time
import uuid

# Initialize LLM
llm = ChatOllama(model="gpt-oss:120b-cloud")

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"

class HumanActionType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    PROVIDE_FEEDBACK = "provide_feedback"
    SKIP = "skip"

class HumanInteraction(BaseModel):
    """Model for human interaction data"""
    interaction_id: str = Field(description="Unique identifier for the interaction")
    action_type: HumanActionType = Field(description="Type of human action")
    feedback: Optional[str] = Field(description="Human feedback or comments")
    timestamp: float = Field(description="Timestamp of the interaction")
    context: Dict[str, Any] = Field(description="Context data for the interaction")

class WorkflowStep(BaseModel):
    """Model for workflow step information"""
    step_id: str = Field(description="Unique identifier for the step")
    step_name: str = Field(description="Human-readable name of the step")
    description: str = Field(description="Description of what this step does")
    requires_approval: bool = Field(description="Whether this step requires human approval")
    approval_criteria: List[str] = Field(description="Criteria for approval")
    status: ApprovalStatus = Field(description="Current status of the step")

class HumanInLoopState(TypedDict):
    """State for human-in-the-loop workflows"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_step: str
    workflow_steps: List[WorkflowStep]
    pending_interactions: List[HumanInteraction]
    completed_interactions: List[HumanInteraction]
    workflow_data: Dict[str, Any]
    human_input: Optional[str]
    waiting_for_human: bool
    interaction_context: Dict[str, Any]

def simulate_human_approval_prompt(step: WorkflowStep, context: Dict[str, Any]) -> str:
    """Generate a prompt for human approval"""
    prompt = f"""
🔍 HUMAN APPROVAL REQUIRED

Step: {step.step_name}
Description: {step.description}

Context:
{json.dumps(context, indent=2)}

Approval Criteria:
{chr(10).join(f"- {criterion}" for criterion in step.approval_criteria)}

Please respond with one of the following:
1. 'approve' - Approve this step
2. 'reject' - Reject this step  
3. 'request_changes: [your feedback]' - Request changes with specific feedback
4. 'skip' - Skip this step (if applicable)

Your response:
"""
    return prompt.strip()

def simulate_human_input(step: WorkflowStep, context: Dict[str, Any]) -> str:
    """Simulate human input for demonstration purposes"""
    print("\n" + "="*60)
    print(simulate_human_approval_prompt(step, context))
    print("="*60)
    
    # In a real implementation, this would wait for actual human input
    # For demonstration, we'll simulate different responses based on the step
    if "data" in step.step_name.lower():
        return "approve"
    elif "security" in step.step_name.lower():
        return "request_changes: Please add encryption for sensitive data"
    elif "deployment" in step.step_name.lower():
        return "approve"
    else:
        return "approve"

def initialize_workflow(state: HumanInLoopState) -> HumanInLoopState:
    """Initialize the workflow with steps that require human interaction"""
    
    workflow_steps = [
        WorkflowStep(
            step_id="data_collection",
            step_name="Data Collection",
            description="Collect and validate input data from various sources",
            requires_approval=True,
            approval_criteria=[
                "Data sources are reliable and up-to-date",
                "Data format is consistent",
                "No missing critical fields",
                "Data privacy requirements are met"
            ],
            status=ApprovalStatus.PENDING
        ),
        WorkflowStep(
            step_id="data_processing",
            step_name="Data Processing",
            description="Process and transform the collected data",
            requires_approval=True,
            approval_criteria=[
                "Processing logic is correct",
                "No data corruption during processing",
                "Output format meets requirements",
                "Performance is acceptable"
            ],
            status=ApprovalStatus.PENDING
        ),
        WorkflowStep(
            step_id="security_review",
            step_name="Security Review",
            description="Review security implications and compliance",
            requires_approval=True,
            approval_criteria=[
                "Security vulnerabilities are addressed",
                "Compliance requirements are met",
                "Access controls are appropriate",
                "Data protection measures are in place"
            ],
            status=ApprovalStatus.PENDING
        ),
        WorkflowStep(
            step_id="final_approval",
            step_name="Final Approval",
            description="Final review and approval before deployment",
            requires_approval=True,
            approval_criteria=[
                "All previous steps are completed successfully",
                "Business requirements are met",
                "Quality standards are satisfied",
                "Stakeholder concerns are addressed"
            ],
            status=ApprovalStatus.PENDING
        )
    ]
    
    state["workflow_steps"] = workflow_steps
    state["current_step"] = workflow_steps[0].step_id
    state["pending_interactions"] = []
    state["completed_interactions"] = []
    state["workflow_data"] = {}
    state["waiting_for_human"] = False
    state["interaction_context"] = {}
    
    print(f"Initialized workflow with {len(workflow_steps)} steps")
    
    return state

def execute_step(state: HumanInLoopState) -> HumanInLoopState:
    """Execute the current workflow step"""
    current_step_id = state["current_step"]
    current_step = next(step for step in state["workflow_steps"] if step.step_id == current_step_id)
    
    print(f"\n🔄 Executing step: {current_step.step_name}")
    
    # Simulate step execution
    execution_data = {
        "step_id": current_step_id,
        "execution_time": time.time(),
        "output": f"Processed data for {current_step.step_name}",
        "metrics": {"accuracy": 0.95, "performance": 0.88}
    }
    
    state["workflow_data"][current_step_id] = execution_data
    state["interaction_context"] = {
        "step": current_step.dict(),
        "execution_data": execution_data
    }
    
    # Mark step as completed and ready for approval
    current_step.status = ApprovalStatus.PENDING
    state["waiting_for_human"] = True
    
    return state

def request_human_input(state: HumanInLoopState) -> HumanInLoopState:
    """Request input from human for approval/review"""
    current_step_id = state["current_step"]
    current_step = next(step for step in state["workflow_steps"] if step.step_id == current_step_id)
    
    # Get human input (simulated)
    human_response = simulate_human_input(current_step, state["interaction_context"])
    
    # Parse human response
    if human_response.lower().startswith('approve'):
        action_type = HumanActionType.APPROVE
        feedback = None
    elif human_response.lower().startswith('reject'):
        action_type = HumanActionType.REJECT
        feedback = human_response
    elif human_response.lower().startswith('request_changes'):
        action_type = HumanActionType.REQUEST_CHANGES
        feedback = human_response.replace('request_changes:', '').strip()
    elif human_response.lower().startswith('skip'):
        action_type = HumanActionType.SKIP
        feedback = None
    else:
        action_type = HumanActionType.PROVIDE_FEEDBACK
        feedback = human_response
    
    # Create interaction record
    interaction = HumanInteraction(
        interaction_id=str(uuid.uuid4()),
        action_type=action_type,
        feedback=feedback,
        timestamp=time.time(),
        context=state["interaction_context"]
    )
    
    state["human_input"] = human_response
    state["pending_interactions"].append(interaction)
    state["waiting_for_human"] = False
    
    print(f"👤 Human Input Received: {action_type}")
    if feedback:
        print(f"   Feedback: {feedback}")
    
    return state

def process_human_input(state: HumanInLoopState) -> HumanInLoopState:
    """Process the human input and update workflow state"""
    if not state["pending_interactions"]:
        return state
    
    latest_interaction = state["pending_interactions"][-1]
    current_step_id = state["current_step"]
    current_step = next(step for step in state["workflow_steps"] if step.step_id == current_step_id)
    
    # Update step status based on human action
    if latest_interaction.action_type == HumanActionType.APPROVE:
        current_step.status = ApprovalStatus.APPROVED
        response = f"✅ Step '{current_step.step_name}' approved by human"
        
    elif latest_interaction.action_type == HumanActionType.REJECT:
        current_step.status = ApprovalStatus.REJECTED
        response = f"❌ Step '{current_step.step_name}' rejected by human"
        
    elif latest_interaction.action_type == HumanActionType.REQUEST_CHANGES:
        current_step.status = ApprovalStatus.REVISION_REQUESTED
        response = f"🔄 Changes requested for '{current_step.step_name}': {latest_interaction.feedback}"
        
    elif latest_interaction.action_type == HumanActionType.SKIP:
        current_step.status = ApprovalStatus.APPROVED  # Skip is treated as approval
        response = f"⏭️ Step '{current_step.step_name}' skipped"
        
    else:
        response = f"📝 Feedback received for '{current_step.step_name}': {latest_interaction.feedback}"
    
    # Move interaction to completed
    state["completed_interactions"].append(latest_interaction)
    state["pending_interactions"].remove(latest_interaction)
    
    # Add response message
    state["messages"].append(AIMessage(content=response))
    
    return state

def determine_next_step(state: HumanInLoopState) -> str:
    """Determine the next step in the workflow"""
    current_step_id = state["current_step"]
    current_step = next(step for step in state["workflow_steps"] if step.step_id == current_step_id)
    
    # Check if current step needs revision
    if current_step.status == ApprovalStatus.REVISION_REQUESTED:
        return "revise_step"
    
    # Check if current step was rejected
    if current_step.status == ApprovalStatus.REJECTED:
        return "handle_rejection"
    
    # Move to next step if current is approved
    if current_step.status == ApprovalStatus.APPROVED:
        current_index = state["workflow_steps"].index(current_step)
        if current_index < len(state["workflow_steps"]) - 1:
            next_step = state["workflow_steps"][current_index + 1]
            state["current_step"] = next_step.step_id
            return "execute_step"
        else:
            return "complete_workflow"
    
    # Default: continue waiting for human input
    return "wait_for_human"

def revise_step(state: HumanInLoopState) -> HumanInLoopState:
    """Revise the current step based on human feedback"""
    current_step_id = state["current_step"]
    current_step = next(step for step in state["workflow_steps"] if step.step_id == current_step_id)
    
    latest_interaction = state["completed_interactions"][-1]
    feedback = latest_interaction.feedback
    
    print(f"\n🔧 Revising step '{current_step.step_name}' based on feedback: {feedback}")
    
    # Simulate revision process
    time.sleep(1)  # Simulate revision time
    
    # Update workflow data with revision
    revision_data = {
        "step_id": current_step_id,
        "revision_time": time.time(),
        "feedback_applied": feedback,
        "revision_output": f"Revised output for {current_step.step_name}"
    }
    
    state["workflow_data"][f"{current_step_id}_revision"] = revision_data
    
    # Reset step status to pending approval
    current_step.status = ApprovalStatus.PENDING
    
    response = f"🔧 Step '{current_step.step_name}' has been revised based on feedback"
    state["messages"].append(AIMessage(content=response))
    
    return state

def handle_rejection(state: HumanInLoopState) -> HumanInLoopState:
    """Handle step rejection"""
    current_step_id = state["current_step"]
    current_step = next(step for step in state["workflow_steps"] if step.step_id == current_step_id)
    
    response = f"🛑 Step '{current_step.step_name}' was rejected. Workflow cannot proceed."
    state["messages"].append(AIMessage(content=response))
    
    return state

def complete_workflow(state: HumanInLoopState) -> HumanInLoopState:
    """Complete the workflow and provide summary"""
    completed_steps = [step for step in state["workflow_steps"] if step.status == ApprovalStatus.APPROVED]
    rejected_steps = [step for step in state["workflow_steps"] if step.status == ApprovalStatus.REJECTED]
    revised_steps = [step for step in state["workflow_steps"] if step.status == ApprovalStatus.REVISION_REQUESTED]
    
    summary = f"""
🎉 WORKFLOW COMPLETED

Summary:
- Total Steps: {len(state['workflow_steps'])}
- Completed: {len(completed_steps)}
- Rejected: {len(rejected_steps)}
- Required Revision: {len(revised_steps)}
- Total Human Interactions: {len(state['completed_interactions'])}

Step Details:
{chr(10).join(f"- {step.step_name}: {step.status.value}" for step in state['workflow_steps'])}

Workflow Data:
{json.dumps(state['workflow_data'], indent=2)}
    """
    
    state["messages"].append(AIMessage(content=summary.strip()))
    
    return state

def create_human_in_loop_graph():
    """Create the human-in-the-loop workflow"""
    
    workflow = StateGraph(HumanInLoopState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_workflow)
    workflow.add_node("execute_step", execute_step)
    workflow.add_node("request_human_input", request_human_input)
    workflow.add_node("process_human_input", process_human_input)
    workflow.add_node("revise_step", revise_step)
    workflow.add_node("handle_rejection", handle_rejection)
    workflow.add_node("complete_workflow", complete_workflow)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "execute_step")
    workflow.add_edge("execute_step", "request_human_input")
    workflow.add_edge("request_human_input", "process_human_input")
    workflow.add_edge("revise_step", "request_human_input")
    workflow.add_edge("handle_rejection", END)
    workflow.add_edge("complete_workflow", END)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "process_human_input",
        determine_next_step,
        {
            "execute_step": "execute_step",
            "revise_step": "revise_step",
            "handle_rejection": "handle_rejection",
            "complete_workflow": "complete_workflow",
            "wait_for_human": "request_human_input"
        }
    )
    
    # Add checkpointing for persistence
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

# Create and test the human-in-the-loop graph
if __name__ == "__main__":
    # Create the graph
    human_loop_graph = create_human_in_loop_graph()
    
    # Display the graph structure
    try:
        display(Image(human_loop_graph.get_graph().draw_mermaid_png()))
    except:
        print("Graph visualization not available")
    
    print("\n" + "="*80)
    print("HUMAN-IN-THE-LOOP WORKFLOW DEMONSTRATION")
    print("="*80)
    
    # Test the workflow
    config = {"configurable": {"thread_id": "human-loop-demo"}}
    
    initial_state = {
        "messages": [HumanMessage(content="Start the approval workflow")]
    }
    
    print("\n🚀 Starting human-in-the-loop workflow...")
    
    # Run the workflow
    result = human_loop_graph.invoke(initial_state, config=config)
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETED")
    print("="*80)
    
    # Display final messages
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(f"\n{message.content}")
    
    print(f"\nFinal State:")
    print(f"- Current Step: {result['current_step']}")
    print(f"- Completed Interactions: {len(result['completed_interactions'])}")
    print(f"- Workflow Steps: {len(result['workflow_steps'])}")
