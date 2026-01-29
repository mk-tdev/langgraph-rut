from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langgraph_viz import visualize
from operator import add

class InputState(TypedDict):
  no_change_value: str
  string_value: Annotated[str, add]
  int_value: Annotated[int, add]
  list_value: Annotated[list[str], add]

def modify_state(state: InputState) -> InputState:
  return state

graph = StateGraph(InputState)
graph.add_node("branch_a", modify_state)
graph.add_node("branch_b", modify_state)
graph.add_edge(START, "branch_a")
graph.add_edge("branch_a", "branch_b")
graph.add_edge("branch_b", END)

graph.set_entry_point("branch_a")

compiled_graph = graph.compile()

display(Image(compiled_graph.get_graph().draw_mermaid_png()))

print("\n🚀 Running Basics with Reducer example with visualization...")

# Use the context manager to run with visualization
with visualize(compiled_graph) as viz_app:
    print("Running with visualization - Browser will open at http://localhost:8765")
    
    # IMPORTANT: Use viz_app, not compiled_graph
    result = viz_app.invoke({"no_change_value": "test", "string_value": "test", "int_value": 1, "list_value": ["test"]})
    print(result)

print("\n✅ Basics with Reducer demonstration completed - Visualization server closed")
