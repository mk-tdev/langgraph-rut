# Instruction for Antigravity: Integrating LangGraph Visualizer

This guide provides step-by-step instructions for integrating functionality to visualize LangGraph workflows using `langgraph-viz`.

## 1. Installation

First, install the visualization package.

```bash
pip install langgraph-viz
```

_Note: If the package is not yet available on PyPI, install it from the source directory provided in the workspace._

## 2. Core Concepts

The visualizer works by wrapping your compiled LangGraph application. This wrapper:

1.  Intercepts execution events (start, end, standard output, state updates).
2.  Starts a local WebSocket server (default port: 8765).
3.  Streams execution data to a real-time browser-based frontend.

**CRITICAL**: You must execute the **wrapped application** instance (`viz_app`), not the original `app`, for the visualization to capture events.

## 3. Integration Patterns

### Pattern A: Context Manager (Recommended)

This is the simplest way to visualize a run. The server starts when entering the block and stops when exiting.

```python
from langgraph_viz import visualize

# ... define and compile your graph ...
app = workflow.compile()

# Visualize specific execution
with visualize(app) as viz_app:
    # IMPORTANT: Use viz_app, not app
    viz_app.stream({"messages": [("user", "Hello world")]})
    # Browser opens automatically at http://localhost:8765
```

### Pattern B: Long-Running Server (Manual Start)

Use this if you want to keep the visualization server running across multiple execution requests or integrate it into a service.

```python
from langgraph_viz import visualize

# ... define and compile your graph ...
app = workflow.compile()

# Initialize visualizer
viz = visualize(app)

# Wrap the app to enable event capturing
viz_app = viz.wrap()

# Run the app (server starts automatically on wrap() or can be explicit)
# viz_app is now a drop-in replacement for app
viz_app.invoke({"query": "First query"})

# Later...
viz_app.invoke({"query": "Second query"})

# Stop server when done (optional, or let process exit)
viz.server.stop()
```

## 4. Usage Example

Here is a complete, minimal example of how to integrate the visualizer into a script.

```python
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph_viz import visualize  # <--- Import

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

def chatbot(state: State):
    return {"messages": [("assistant", "I am a visualized bot!")]}

workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)

app = workflow.compile()

# --- Visualization Integration ---
if __name__ == "__main__":
    # Use the context manager to run with visualization
    with visualize(app) as viz_app:
        print("Running with visualization...")

        # Execute the workflow using the wrapped app
        viz_app.invoke({"messages": [("user", "Hi")]})

        input("Press Enter to close...")
```

## 5. Troubleshooting checklist

- **Browser doesn't open?** Check the console output. You can manually visit `http://localhost:8765`.
- **No data in visualizer?** Ensure you are calling `.invoke()` or `.stream()` on the **wrapped** `viz_app` object, not the original `app`.
- **Port conflict?** Change the port using `visualize(app, port=8766)`.
