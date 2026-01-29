# Advanced LangGraph Concepts

This folder contains comprehensive examples of advanced LangGraph concepts and patterns. Each example demonstrates sophisticated workflow patterns that go beyond basic graph construction.

## 📁 File Structure

```
advanced/
├── README.md                          # This file
├── conditional_routing.py              # Advanced conditional routing patterns
├── parallel_execution.py               # Parallel execution and fan-out/fan-in
├── human_in_the_loop.py                # Human-in-the-loop workflows
├── subgraphs.py                        # Subgraph architecture and composition
├── streaming_checkpointing.py          # Real-time streaming and checkpointing
├── dynamic_graph_modification.py       # Runtime graph modification
├── custom_state_management.py          # Advanced state management
└── error_handling_recovery.py          # Error handling and recovery patterns
```

## 🚀 Advanced Concepts Covered

### 1. **Conditional Routing** (`conditional_routing.py`)
- **Multi-way conditional routing** based on complex state analysis
- **Dynamic routing** with multiple criteria (intent, sentiment, urgency, complexity)
- **Nested conditional logic** with secondary routing decisions
- **Message content analysis** for intelligent routing

**Key Features:**
- Intent classification (question, command, conversation, request)
- Sentiment analysis (positive, negative, neutral)
- Urgency detection (high, medium, low)
- Complexity scoring and topic categorization
- Multi-level routing with primary and secondary decision points

### 2. **Parallel Execution** (`parallel_execution.py`)
- **Parallel processing** of multiple tasks using ThreadPoolExecutor
- **Fan-out/fan-in patterns** for concurrent operations
- **Dynamic task creation** based on input analysis
- **Performance monitoring** and efficiency calculations

**Key Features:**
- Concurrent API calls and data processing
- Parallel analysis and synthesis
- Task result aggregation
- Performance metrics and speedup calculations
- Error handling in parallel operations

### 3. **Human-in-the-Loop** (`human_in_the_loop.py`)
- **Approval workflows** with human confirmation
- **Interactive decision points** and breakpoints
- **Manual review and correction** processes
- **Feedback incorporation** loops

**Key Features:**
- Workflow step approval system
- Human interaction tracking
- Revision and correction workflows
- Checkpointing for human intervention
- Approval criteria and context management

### 4. **Subgraphs** (`subgraphs.py`)
- **Reusable subgraph** components
- **Nested subgraph** architectures
- **State management** between parent and child graphs
- **Dynamic subgraph selection** and composition

**Key Features:**
- Modular graph design
- Data processing subgraph
- Content analysis subgraph
- Quality check subgraph
- Report generation subgraph
- State passing between graphs

### 5. **Streaming & Checkpointing** (`streaming_checkpointing.py`)
- **Real-time streaming** of graph execution
- **State persistence** and recovery
- **Checkpoint management** and rollback
- **Resumable workflows** with state restoration

**Key Features:**
- Event streaming system
- Automatic and manual checkpoints
- State snapshot management
- Recovery from checkpoints
- Progress tracking and intermediate results

### 6. **Dynamic Graph Modification** (`dynamic_graph_modification.py`)
- **Runtime node addition** and removal
- **Dynamic edge creation** and deletion
- **Adaptive workflow** evolution
- **Runtime graph reconfiguration**

**Key Features:**
- Dynamic node registration
- Runtime graph restructuring
- Adaptive routing based on performance
- Graph modification history
- Evolution over multiple iterations

### 7. **Custom State Management** (`custom_state_management.py`)
- **Custom state schemas** and validation
- **State transformation** and normalization
- **Multi-level state** hierarchies
- **State persistence** and retrieval

**Key Features:**
- Advanced state validation
- State transformation pipelines
- State snapshots and versioning
- Custom state reducers
- State metadata management

### 8. **Error Handling & Recovery** (`error_handling_recovery.py`)
- **Multiple error types** and handling strategies
- **Automatic retry** with exponential backoff
- **Circuit breaker** patterns
- **Graceful degradation** and fallback strategies

**Key Features:**
- Comprehensive error classification
- Retry mechanisms with backoff
- Circuit breaker implementation
- Fallback strategies
- System health monitoring
- Recovery workflow automation

## 🛠️ Technical Requirements

All examples use the following core dependencies:

```python
langgraph>=1.0.1
langchain-core>=1.0.1
langchain-ollama
pydantic>=2.0
```

Additional requirements for specific examples:
- `parallel_execution.py`: `concurrent.futures` (built-in)
- `streaming_checkpointing.py`: `asyncio`, `threading`, `queue` (built-in)
- `custom_state_management.py`: `pickle`, `hashlib` (built-in)

## 📖 Usage Examples

### Running Individual Examples

Each example can be run independently:

```bash
# Run conditional routing example
python conditional_routing.py

# Run parallel execution example
python parallel_execution.py

# Run human-in-the-loop example
python human_in_the_loop.py
```

### Integration with Existing Workflows

These patterns can be integrated into your existing LangGraph workflows:

```python
from advanced.conditional_routing import create_advanced_routing_graph
from advanced.error_handling_recovery import create_error_handling_graph

# Create advanced routing
routing_graph = create_advanced_routing_graph()

# Create error handling workflow
error_graph = create_error_handling_graph()
```

## 🎯 Learning Path

### Beginner → Advanced Progression

1. **Start with Basic Concepts**: Ensure you understand basic LangGraph patterns
2. **Conditional Routing**: Learn intelligent decision-making in workflows
3. **Parallel Execution**: Master concurrent processing patterns
4. **Error Handling**: Implement robust error management
5. **State Management**: Advanced state handling techniques
6. **Human-in-the-Loop**: Add human interaction capabilities
7. **Streaming**: Real-time workflow monitoring
8. **Subgraphs**: Modular architecture design
9. **Dynamic Modification**: Runtime workflow adaptation

### Key Concepts to Master

- **State Management**: How data flows through your graphs
- **Conditional Logic**: Making intelligent routing decisions
- **Error Resilience**: Building robust workflows
- **Modularity**: Creating reusable components
- **Performance**: Optimizing for speed and efficiency
- **Monitoring**: Tracking and debugging workflows

## 🔧 Customization Guidelines

### Adapting Patterns to Your Use Case

1. **Modify State Schemas**: Adjust the TypedDict classes to match your data
2. **Custom Routing Logic**: Implement domain-specific routing rules
3. **Add Custom Nodes**: Create nodes for your specific business logic
4. **Configure Error Handling**: Set up error types and recovery strategies
5. **Integrate External Services**: Add API calls and database operations

### Best Practices

1. **Start Simple**: Begin with basic patterns and add complexity gradually
2. **Test Thoroughly**: Each pattern includes comprehensive test scenarios
3. **Monitor Performance**: Use built-in metrics to optimize your workflows
4. **Handle Errors Gracefully**: Implement proper error handling from the start
5. **Document Your Work**: Follow the documentation patterns shown in examples

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **State Validation**: Check that your state schemas match your data
3. **Graph Compilation**: Verify all nodes and edges are properly defined
4. **Memory Usage**: Monitor state size in long-running workflows
5. **Performance**: Use parallel execution for CPU-intensive tasks

### Debugging Tips

1. **Use Graph Visualization**: All examples include graph drawing capabilities
2. **Enable Logging**: Add print statements to track execution flow
3. **Check State**: Use state inspection to debug data flow issues
4. **Test Components**: Test individual nodes before integrating
5. **Monitor Errors**: Use the error handling patterns to catch issues early

## 📚 Additional Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Python Concurrency Documentation](https://docs.python.org/3/library/concurrency.html)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)

## 🤝 Contributing

Feel free to extend these examples with:
- New advanced patterns
- Additional use cases
- Performance optimizations
- Better error handling
- Enhanced documentation

## 📄 License

These examples are provided for educational purposes. Please ensure compliance with the licenses of all dependencies used.
