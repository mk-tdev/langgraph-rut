"""
Advanced Error Handling and Recovery in LangGraph

This example demonstrates comprehensive error handling and recovery patterns including:
- Multiple error types and handling strategies
- Automatic retry mechanisms with exponential backoff
- Circuit breaker patterns for failing services
- Graceful degradation and fallback strategies
- Error recovery workflows and state restoration
"""

import time
import random
import json
from typing import Literal, TypedDict, Annotated, Sequence, List, Dict, Any, Optional, Callable
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from pydantic import BaseModel, Field
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Initialize LLM
llm = ChatOllama(model="gpt-oss:120b-cloud")

class ErrorType(str, Enum):
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"
    PERMISSION_ERROR = "permission_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(str, Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    MANUAL_INTERVENTION = "manual_intervention"
    SKIP = "skip"
    ABORT = "abort"

@dataclass
class ErrorInfo:
    """Detailed error information"""
    error_id: str = field(default_factory=lambda: f"error_{int(time.time())}")
    error_type: ErrorType = ErrorType.UNKNOWN_ERROR
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    message: str = ""
    original_exception: Optional[Exception] = None
    stack_trace: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    node_name: str = ""
    recovery_attempted: bool = False
    recovery_successful: bool = False

@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[Exception] = field(default_factory=list)

@dataclass
class CircuitBreakerState:
    """Circuit breaker state"""
    failure_count: int = 0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    last_failure_time: float = 0.0
    state: Literal["CLOSED", "OPEN", "HALF_OPEN"] = "CLOSED"

class ErrorRecoveryState(TypedDict):
    """State for error handling and recovery workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_node: str
    errors: List[ErrorInfo]
    recovery_strategies: Dict[str, RecoveryStrategy]
    circuit_breakers: Dict[str, CircuitBreakerState]
    retry_configs: Dict[str, RetryConfig]
    fallback_data: Dict[str, Any]
    recovery_history: List[Dict[str, Any]]
    system_health: Dict[str, bool]
    degradation_level: float  # 0.0 = full functionality, 1.0 = minimal functionality

class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self):
        self.error_patterns = {}
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        self.retry_configs = {}
    
    def classify_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Classify and create detailed error information"""
        context = context or {}
        
        # Determine error type based on exception
        error_type = ErrorType.UNKNOWN_ERROR
        message = str(exception)
        
        if "network" in message.lower() or "connection" in message.lower():
            error_type = ErrorType.NETWORK_ERROR
        elif "timeout" in message.lower():
            error_type = ErrorType.TIMEOUT_ERROR
        elif "permission" in message.lower() or "access" in message.lower():
            error_type = ErrorType.PERMISSION_ERROR
        elif "validation" in message.lower() or "invalid" in message.lower():
            error_type = ErrorType.VALIDATION_ERROR
        elif "api" in message.lower():
            error_type = ErrorType.API_ERROR
        elif "resource" in message.lower():
            error_type = ErrorType.RESOURCE_ERROR
        
        # Determine severity
        severity = ErrorSeverity.MEDIUM
        if error_type in [ErrorType.CRITICAL, ErrorType.PERMISSION_ERROR]:
            severity = ErrorSeverity.HIGH
        elif error_type == ErrorType.NETWORK_ERROR:
            severity = ErrorSeverity.LOW
        
        return ErrorInfo(
            error_type=error_type,
            severity=severity,
            message=message,
            original_exception=exception,
            stack_trace=traceback.format_exc(),
            context=context,
            node_name=context.get("node_name", "unknown")
        )
    
    def should_retry(self, error_info: ErrorInfo, retry_config: RetryConfig) -> bool:
        """Determine if error should be retried"""
        if error_info.retry_count >= retry_config.max_retries:
            return False
        
        # Check if exception type is in retry list
        if retry_config.retry_on_exceptions:
            return any(isinstance(error_info.original_exception, exc_type) 
                      for exc_type in retry_config.retry_on_exceptions)
        
        # Default retry logic based on error type
        retryable_types = [
            ErrorType.NETWORK_ERROR,
            ErrorType.TIMEOUT_ERROR,
            ErrorType.API_ERROR
        ]
        
        return error_info.error_type in retryable_types
    
    def calculate_retry_delay(self, retry_count: int, retry_config: RetryConfig) -> float:
        """Calculate retry delay with exponential backoff and jitter"""
        delay = retry_config.base_delay * (retry_config.exponential_base ** retry_count)
        delay = min(delay, retry_config.max_delay)
        
        if retry_config.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def check_circuit_breaker(self, service_name: str, circuit_breaker: CircuitBreakerState) -> bool:
        """Check if circuit breaker allows execution"""
        current_time = time.time()
        
        if circuit_breaker.state == "OPEN":
            # Check if recovery timeout has passed
            if current_time - circuit_breaker.last_failure_time > circuit_breaker.recovery_timeout:
                circuit_breaker.state = "HALF_OPEN"
                print(f"🔄 Circuit breaker for {service_name} transitioning to HALF_OPEN")
                return True
            else:
                print(f"🚫 Circuit breaker for {service_name} is OPEN")
                return False
        
        elif circuit_breaker.state == "HALF_OPEN":
            # Allow one request to test the waters
            return True
        
        else:  # CLOSED
            return True
    
    def record_success(self, service_name: str, circuit_breaker: CircuitBreakerState):
        """Record successful operation"""
        if circuit_breaker.state == "HALF_OPEN":
            circuit_breaker.state = "CLOSED"
            circuit_breaker.failure_count = 0
            print(f"✅ Circuit breaker for {service_name} reset to CLOSED")
        elif circuit_breaker.state == "CLOSED":
            # Reset failure count on success
            circuit_breaker.failure_count = 0
    
    def record_failure(self, service_name: str, circuit_breaker: CircuitBreakerState):
        """Record failed operation"""
        circuit_breaker.failure_count += 1
        circuit_breaker.last_failure_time = time.time()
        
        if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
            if circuit_breaker.state != "OPEN":
                circuit_breaker.state = "OPEN"
                print(f"🚫 Circuit breaker for {service_name} opened due to {circuit_breaker.failure_count} failures")

# Global error handler
error_handler = ErrorHandler()

def resilient_api_call(state: ErrorRecoveryState, service_name: str, 
                     retry_config: RetryConfig = None, 
                     circuit_breaker: CircuitBreakerState = None) -> Dict[str, Any]:
    """Make a resilient API call with error handling"""
    
    if retry_config is None:
        retry_config = RetryConfig(max_retries=3, base_delay=1.0)
    
    if circuit_breaker is None:
        circuit_breaker = CircuitBreakerState(failure_threshold=3, recovery_timeout=30.0)
    
    # Check circuit breaker
    if not error_handler.check_circuit_breaker(service_name, circuit_breaker):
        error_info = ErrorInfo(
            error_type=ErrorType.NETWORK_ERROR,
            severity=ErrorSeverity.HIGH,
            message=f"Circuit breaker OPEN for {service_name}",
            context={"service": service_name, "circuit_breaker": True}
        )
        state["errors"].append(error_info)
        return {"success": False, "error": "Circuit breaker open"}
    
    # Attempt API call with retries
    for attempt in range(retry_config.max_retries + 1):
        try:
            # Simulate API call
            if random.random() < 0.3:  # 30% failure rate
                raise Exception(f"Simulated API failure for {service_name}")
            
            # Simulate processing time
            time.sleep(0.1)
            
            # Record success
            error_handler.record_success(service_name, circuit_breaker)
            
            return {
                "success": True,
                "data": f"API response from {service_name}",
                "attempt": attempt + 1
            }
            
        except Exception as e:
            error_info = error_handler.classify_error(e, {"node_name": service_name})
            error_info.retry_count = attempt
            
            if attempt < retry_config.max_retries and error_handler.should_retry(error_info, retry_config):
                delay = error_handler.calculate_retry_delay(attempt, retry_config)
                print(f"🔄 Retrying {service_name} in {delay:.2f}s (attempt {attempt + 1}/{retry_config.max_retries + 1})")
                time.sleep(delay)
                continue
            else:
                # Record failure and break
                error_handler.record_failure(service_name, circuit_breaker)
                state["errors"].append(error_info)
                return {"success": False, "error": str(e), "attempts": attempt + 1}

def data_processing_with_error_handling(state: ErrorRecoveryState) -> ErrorRecoveryState:
    """Data processing node with comprehensive error handling"""
    node_name = "data_processing"
    state["current_node"] = node_name
    
    print(f"🔄 Starting {node_name}")
    
    try:
        # Simulate data processing that might fail
        if random.random() < 0.2:  # 20% failure rate
            raise ValueError("Data validation failed: Invalid format detected")
        
        # Simulate processing time
        time.sleep(0.5)
        
        result = {
            "processed_records": 1000,
            "processing_time": 0.5,
            "quality_score": 0.95
        }
        
        state["messages"].append(AIMessage(content=f"✅ {node_name} completed successfully"))
        state["system_health"][node_name] = True
        
        print(f"✅ {node_name} completed: {result}")
        
    except Exception as e:
        error_info = error_handler.classify_error(e, {"node_name": node_name})
        state["errors"].append(error_info)
        state["system_health"][node_name] = False
        
        print(f"❌ {node_name} failed: {e}")
        
        # Add error message
        state["messages"].append(AIMessage(content=f"❌ {node_name} failed: {e}"))
    
    return state

def api_integration_with_circuit_breaker(state: ErrorRecoveryState) -> ErrorRecoveryState:
    """API integration node with circuit breaker pattern"""
    node_name = "api_integration"
    state["current_node"] = node_name
    
    print(f"🔄 Starting {node_name}")
    
    # Set up circuit breaker
    if node_name not in state["circuit_breakers"]:
        state["circuit_breakers"][node_name] = CircuitBreakerState(
            failure_threshold=3,
            recovery_timeout=30.0
        )
    
    circuit_breaker = state["circuit_breakers"][node_name]
    
    # Make resilient API call
    result = resilient_api_call(state, "external_api", circuit_breaker=circuit_breaker)
    
    if result["success"]:
        state["messages"].append(AIMessage(content=f"✅ {node_name} completed successfully"))
        state["system_health"][node_name] = True
        print(f"✅ {node_name} completed: {result['data']}")
    else:
        state["messages"].append(AIMessage(content=f"❌ {node_name} failed: {result['error']}"))
        state["system_health"][node_name] = False
        print(f"❌ {node_name} failed: {result['error']}")
    
    return state

def validation_with_retry(state: ErrorRecoveryState) -> ErrorRecoveryState:
    """Validation node with retry mechanism"""
    node_name = "validation"
    state["current_node"] = node_name
    
    print(f"🔄 Starting {node_name}")
    
    # Set up retry configuration
    if node_name not in state["retry_configs"]:
        state["retry_configs"][node_name] = RetryConfig(
            max_retries=3,
            base_delay=0.5,
            exponential_base=2.0,
            jitter=True
        )
    
    retry_config = state["retry_configs"][node_name]
    
    # Attempt validation with retries
    for attempt in range(retry_config.max_retries + 1):
        try:
            # Simulate validation that might fail temporarily
            if random.random() < 0.4 and attempt < retry_config.max_retries:  # 40% failure rate, but succeed on last attempt
                raise Exception(f"Validation service temporarily unavailable (attempt {attempt + 1})")
            
            # Simulate validation
            time.sleep(0.2)
            
            result = {
                "validation_passed": True,
                "validation_time": 0.2,
                "rules_checked": 25
            }
            
            state["messages"].append(AIMessage(content=f"✅ {node_name} completed successfully"))
            state["system_health"][node_name] = True
            print(f"✅ {node_name} completed: {result}")
            break
            
        except Exception as e:
            if attempt < retry_config.max_retries:
                delay = error_handler.calculate_retry_delay(attempt, retry_config)
                print(f"🔄 Retrying {node_name} in {delay:.2f}s (attempt {attempt + 1}/{retry_config.max_retries + 1})")
                time.sleep(delay)
            else:
                error_info = error_handler.classify_error(e, {"node_name": node_name})
                error_info.retry_count = attempt
                state["errors"].append(error_info)
                state["system_health"][node_name] = False
                
                state["messages"].append(AIMessage(content=f"❌ {node_name} failed after {attempt + 1} attempts: {e}"))
                print(f"❌ {node_name} failed after {attempt + 1} attempts: {e}")
    
    return state

def fallback_handler(state: ErrorRecoveryState) -> ErrorRecoveryState:
    """Fallback handler for graceful degradation"""
    node_name = "fallback_handler"
    state["current_node"] = node_name
    
    print(f"🔄 Starting {node_name}")
    
    # Analyze system health
    failed_nodes = [node for node, healthy in state["system_health"].items() if not healthy]
    
    if not failed_nodes:
        state["messages"].append(AIMessage(content="✅ All systems healthy, no fallback needed"))
        print("✅ All systems healthy, no fallback needed")
        return state
    
    print(f"🚨 Detected failed nodes: {failed_nodes}")
    
    # Apply fallback strategies
    fallback_results = {}
    
    for failed_node in failed_nodes:
        if failed_node == "data_processing":
            # Fallback data processing
            fallback_results[failed_node] = {
                "fallback_used": "simplified_processing",
                "processed_records": 500,  # Reduced capacity
                "quality_score": 0.85  # Lower quality
            }
            state["degradation_level"] += 0.2
            
        elif failed_node == "api_integration":
            # Fallback API integration
            fallback_results[failed_node] = {
                "fallback_used": "cached_data",
                "data_age": "1 hour old",
                "completeness": 0.7
            }
            state["degradation_level"] += 0.3
            
        elif failed_node == "validation":
            # Fallback validation
            fallback_results[failed_node] = {
                "fallback_used": "basic_validation",
                "rules_checked": 10,  # Reduced validation
                "confidence": 0.8
            }
            state["degradation_level"] += 0.1
    
    # Store fallback data
    state["fallback_data"] = fallback_results
    
    # Update recovery history
    recovery_entry = {
        "timestamp": time.time(),
        "failed_nodes": failed_nodes,
        "fallback_strategies": list(fallback_results.keys()),
        "degradation_level": state["degradation_level"]
    }
    state["recovery_history"].append(recovery_entry)
    
    state["messages"].append(AIMessage(
        content=f"🔄 Applied fallback strategies for {len(failed_nodes)} failed nodes. "
               f"System degradation level: {state['degradation_level']:.1f}"
    ))
    
    print(f"🔄 Applied fallback strategies: {list(fallback_results.keys())}")
    print(f"📊 System degradation level: {state['degradation_level']:.1f}")
    
    return state

def error_analyzer(state: ErrorRecoveryState) -> ErrorRecoveryState:
    """Analyze errors and determine recovery strategies"""
    node_name = "error_analyzer"
    state["current_node"] = node_name
    
    print(f"🔄 Starting {node_name}")
    
    if not state["errors"]:
        state["messages"].append(AIMessage(content="✅ No errors to analyze"))
        print("✅ No errors to analyze")
        return state
    
    # Analyze error patterns
    error_types = {}
    error_severities = {}
    failed_nodes = set()
    
    for error in state["errors"]:
        error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        error_severities[error.severity] = error_severities.get(error.severity, 0) + 1
        failed_nodes.add(error.node_name)
    
    # Determine recovery strategies
    recovery_strategies = {}
    
    for error_type, count in error_types.items():
        if error_type == ErrorType.NETWORK_ERROR and count > 2:
            recovery_strategies["network"] = RecoveryStrategy.RETRY
        elif error_type == ErrorType.API_ERROR and count > 1:
            recovery_strategies["api"] = RecoveryStrategy.CIRCUIT_BREAKER
        elif error_type == ErrorType.VALIDATION_ERROR:
            recovery_strategies["validation"] = RecoveryStrategy.FALLBACK
        elif error_type in [ErrorType.PERMISSION_ERROR, ErrorType.RESOURCE_ERROR]:
            recovery_strategies["critical"] = RecoveryStrategy.MANUAL_INTERVENTION
    
    state["recovery_strategies"] = recovery_strategies
    
    # Create analysis report
    analysis_report = {
        "total_errors": len(state["errors"]),
        "error_types": error_types,
        "error_severities": error_severities,
        "failed_nodes": list(failed_nodes),
        "recovery_strategies": {k: v.value for k, v in recovery_strategies.items()}
    }
    
    state["messages"].append(AIMessage(
        content=f"📊 Error analysis completed: {len(state['errors'])} errors detected. "
               f"Recovery strategies determined for {len(recovery_strategies)} error categories."
    ))
    
    print(f"📊 Error analysis:")
    print(f"- Total errors: {len(state['errors'])}")
    print(f"- Error types: {error_types}")
    print(f"- Failed nodes: {list(failed_nodes)}")
    print(f"- Recovery strategies: {list(recovery_strategies.keys())}")
    
    return state

def recovery_executor(state: ErrorRecoveryState) -> ErrorRecoveryState:
    """Execute recovery strategies based on analysis"""
    node_name = "recovery_executor"
    state["current_node"] = node_name
    
    print(f"🔄 Starting {node_name}")
    
    recovery_strategies = state.get("recovery_strategies", {})
    
    if not recovery_strategies:
        state["messages"].append(AIMessage(content="✅ No recovery strategies to execute"))
        print("✅ No recovery strategies to execute")
        return state
    
    executed_strategies = []
    
    for category, strategy in recovery_strategies.items():
        print(f"🔧 Executing {strategy.value} strategy for {category}")
        
        if strategy == RecoveryStrategy.RETRY:
            # Implement retry logic
            executed_strategies.append(f"Retry configured for {category}")
            
        elif strategy == RecoveryStrategy.FALLBACK:
            # Implement fallback logic
            executed_strategies.append(f"Fallback activated for {category}")
            
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            # Implement circuit breaker logic
            executed_strategies.append(f"Circuit breaker enabled for {category}")
            
        elif strategy == RecoveryStrategy.MANUAL_INTERVENTION:
            # Flag for manual intervention
            executed_strategies.append(f"Manual intervention required for {category}")
        
        # Simulate recovery execution time
        time.sleep(0.3)
    
    # Record recovery execution
    recovery_entry = {
        "timestamp": time.time(),
        "strategies_executed": executed_strategies,
        "success": True
    }
    state["recovery_history"].append(recovery_entry)
    
    state["messages"].append(AIMessage(
        content=f"🔧 Executed {len(executed_strategies)} recovery strategies"
    ))
    
    print(f"🔧 Recovery execution completed: {len(executed_strategies)} strategies")
    
    return state

def system_health_monitor(state: ErrorRecoveryState) -> ErrorRecoveryState:
    """Monitor overall system health and determine final status"""
    node_name = "system_health_monitor"
    state["current_node"] = node_name
    
    print(f"🔄 Starting {node_name}")
    
    # Calculate system health metrics
    total_nodes = len(state["system_health"])
    healthy_nodes = sum(1 for healthy in state["system_health"].values() if healthy)
    health_percentage = (healthy_nodes / total_nodes) * 100 if total_nodes > 0 else 0
    
    # Calculate error metrics
    total_errors = len(state["errors"])
    critical_errors = sum(1 for error in state["errors"] if error.severity == ErrorSeverity.CRITICAL)
    
    # Determine overall system status
    if health_percentage >= 90 and critical_errors == 0:
        system_status = "HEALTHY"
        status_emoji = "✅"
    elif health_percentage >= 70 and critical_errors == 0:
        system_status = "DEGRADED"
        status_emoji = "⚠️"
    else:
        system_status = "CRITICAL"
        status_emoji = "🚨"
    
    # Create health report
    health_report = {
        "system_status": system_status,
        "health_percentage": health_percentage,
        "healthy_nodes": healthy_nodes,
        "total_nodes": total_nodes,
        "total_errors": total_errors,
        "critical_errors": critical_errors,
        "degradation_level": state["degradation_level"],
        "recovery_attempts": len(state["recovery_history"])
    }
    
    state["messages"].append(AIMessage(
        content=f"{status_emoji} System Health Report:\n"
               f"- Status: {system_status}\n"
               f"- Health: {health_percentage:.1f}% ({healthy_nodes}/{total_nodes} nodes)\n"
               f"- Errors: {total_errors} total, {critical_errors} critical\n"
               f"- Degradation: {state['degradation_level']:.1f}\n"
               f"- Recovery attempts: {len(state['recovery_history'])}"
    ))
    
    print(f"📊 System Health: {system_status} ({health_percentage:.1f}% healthy)")
    
    return state

def create_error_handling_graph():
    """Create the error handling and recovery workflow"""
    
    workflow = StateGraph(ErrorRecoveryState)
    
    # Add nodes
    workflow.add_node("data_processing", data_processing_with_error_handling)
    workflow.add_node("api_integration", api_integration_with_circuit_breaker)
    workflow.add_node("validation", validation_with_retry)
    workflow.add_node("error_analyzer", error_analyzer)
    workflow.add_node("fallback_handler", fallback_handler)
    workflow.add_node("recovery_executor", recovery_executor)
    workflow.add_node("system_health_monitor", system_health_monitor)
    
    # Add edges
    workflow.add_edge(START, "data_processing")
    workflow.add_edge("data_processing", "api_integration")
    workflow.add_edge("api_integration", "validation")
    workflow.add_edge("validation", "error_analyzer")
    workflow.add_edge("error_analyzer", "fallback_handler")
    workflow.add_edge("fallback_handler", "recovery_executor")
    workflow.add_edge("recovery_executor", "system_health_monitor")
    workflow.add_edge("system_health_monitor", END)
    
    return workflow.compile()

def demonstrate_error_handling_scenarios():
    """Demonstrate various error handling scenarios"""
    print("\n" + "="*80)
    print("ERROR HANDLING SCENARIOS DEMONSTRATION")
    print("="*80)
    
    # Create graph
    graph = create_error_handling_graph()
    
    # Test different scenarios
    scenarios = [
        {
            "name": "Normal Operation",
            "description": "All nodes work correctly",
            "seed": 42  # Fixed seed for reproducible results
        },
        {
            "name": "Random Failures",
            "description": "Random failures in different nodes",
            "seed": None  # Random seed
        },
        {
            "name": "High Failure Rate",
            "description": "Simulate high failure rate",
            "seed": 123
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        
        # Set random seed if specified
        if scenario['seed'] is not None:
            random.seed(scenario['seed'])
        
        # Set up initial state
        initial_state = {
            "messages": [HumanMessage(content=f"Start error handling scenario: {scenario['name']}")],
            "current_node": "",
            "errors": [],
            "recovery_strategies": {},
            "circuit_breakers": {},
            "retry_configs": {},
            "fallback_data": {},
            "recovery_history": [],
            "system_health": {},
            "degradation_level": 0.0
        }
        
        config = {"configurable": {"thread_id": f"error-scenario-{i}"}}
        
        # Run the scenario
        result = graph.invoke(initial_state, config=config)
        
        # Display results
        print(f"\n📊 Scenario Results:")
        print(f"- Total errors: {len(result['errors'])}")
        print(f"- Recovery attempts: {len(result['recovery_history'])}")
        print(f"- System degradation: {result['degradation_level']:.1f}")
        print(f"- System health: {sum(1 for h in result['system_health'].values() if h)}/{len(result['system_health'])} nodes healthy")
        
        # Display error summary
        if result['errors']:
            error_types = {}
            for error in result['errors']:
                error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            print(f"- Error types: {error_types}")
        
        # Reset random seed
        random.seed()

def demonstrate_circuit_breaker_pattern():
    """Demonstrate circuit breaker pattern specifically"""
    print("\n" + "="*80)
    print("CIRCUIT BREAKER PATTERN DEMONSTRATION")
    print("="*80)
    
    # Create circuit breaker
    circuit_breaker = CircuitBreakerState(
        failure_threshold=3,
        recovery_timeout=5.0  # Short timeout for demo
    )
    
    state = {
        "messages": [],
        "current_node": "",
        "errors": [],
        "recovery_strategies": {},
        "circuit_breakers": {"demo_service": circuit_breaker},
        "retry_configs": {},
        "fallback_data": {},
        "recovery_history": [],
        "system_health": {},
        "degradation_level": 0.0
    }
    
    print(f"\n🔄 Testing circuit breaker with failure threshold of {circuit_breaker.failure_threshold}")
    
    # Simulate multiple failures
    for i in range(6):
        print(f"\n--- Attempt {i + 1} ---")
        
        result = resilient_api_call(state, "demo_service", circuit_breaker=circuit_breaker)
        
        if result["success"]:
            print(f"✅ Success: {result['data']}")
        else:
            print(f"❌ Failure: {result['error']}")
        
        print(f"Circuit breaker state: {circuit_breaker.state}")
        print(f"Failure count: {circuit_breaker.failure_count}")
        
        # Wait for recovery timeout if circuit is open
        if circuit_breaker.state == "OPEN":
            print(f"⏳ Waiting {circuit_breaker.recovery_timeout}s for recovery timeout...")
            time.sleep(circuit_breaker.recovery_timeout + 0.1)

def demonstrate_graceful_degradation():
    """Demonstrate graceful degradation"""
    print("\n" + "="*80)
    print("GRACEFUL DEGRADATION DEMONSTRATION")
    print("="*80)
    
    # Create graph
    graph = create_error_handling_graph()
    
    # Set up state with multiple failures
    initial_state = {
        "messages": [HumanMessage(content="Test graceful degradation with multiple failures")],
        "current_node": "",
        "errors": [],
        "recovery_strategies": {},
        "circuit_breakers": {},
        "retry_configs": {},
        "fallback_data": {},
        "recovery_history": [],
        "system_health": {},
        "degradation_level": 0.0
    }
    
    # Force failures by setting random seed to ensure failures
    random.seed(999)  # Seed that causes multiple failures
    
    config = {"configurable": {"thread_id": "graceful-degradation-demo"}}
    
    print(f"\n🔄 Running workflow with forced failures to test graceful degradation...")
    
    # Run the workflow
    result = graph.invoke(initial_state, config=config)
    
    print(f"\n📊 Graceful Degradation Results:")
    print(f"- Final degradation level: {result['degradation_level']:.1f}")
    print(f"- Fallback strategies used: {list(result['fallback_data'].keys()) if result['fallback_data'] else 'None'}")
    print(f"- Recovery history entries: {len(result['recovery_history'])}")
    
    # Display fallback details
    if result['fallback_data']:
        print(f"\n🔄 Fallback Details:")
        for node, fallback_info in result['fallback_data'].items():
            print(f"- {node}: {fallback_info}")

# Main execution
if __name__ == "__main__":
    # Create the graph
    error_handling_graph = create_error_handling_graph()
    
    # Display the graph structure
    try:
        display(Image(error_handling_graph.get_graph().draw_mermaid_png()))
    except:
        print("Graph visualization not available")
    
    print("\n" + "="*80)
    print("ADVANCED ERROR HANDLING AND RECOVERY DEMONSTRATIONS")
    print("="*80)
    
    # Demonstrate error handling scenarios
    demonstrate_error_handling_scenarios()
    
    # Demonstrate circuit breaker pattern
    demonstrate_circuit_breaker_pattern()
    
    # Demonstrate graceful degradation
    demonstrate_graceful_degradation()
    
    print("\n" + "="*80)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("="*80)
    
    # Final summary
    print(f"\n📊 Final Error Handler Summary:")
    print(f"- Error handler initialized: {error_handler is not None}")
    print(f"- Circuit breakers managed: {len(error_handler.circuit_breakers)}")
    print(f"- Recovery strategies available: {len(RecoveryStrategy)}")
    print(f"- Error types supported: {len(ErrorType)}")
