from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langgraph_viz import visualize
from operator import add

class InputState(TypedDict):
  name: str
  age: int

def modify_state(state: InputState) -> InputState:
  print(f"Current state modify: {state}")
  state["age"] += 1
  print(f"Updated state modify: {state}")
  return state

def router(state: InputState) -> str:
  print(f"Current state router: {state}")
  if state["age"] < 30:
    print("Going to branch_a")
    return "branch_a"
  print("Ending graph")
  return "__end__"

graph = StateGraph(InputState)

graph.add_node("branch_a", modify_state)
graph.add_node("branch_b", modify_state)
graph.add_edge(START, "branch_a")
graph.add_edge("branch_a", "branch_b")
graph.add_conditional_edges("branch_b", router, {"branch_a": "branch_a", "__end__": END})
# graph.add_edge("branch_b", END)

graph.set_entry_point("branch_a")

compiled_graph = graph.compile()

# with visualize(compiled_graph) as viz_app:
#   result = viz_app.invoke({"name": "John", "age": 30})

display(Image(compiled_graph.get_graph().draw_mermaid_png()))

result = compiled_graph.invoke({"name": "John", "age": 25})
print(result)

## Reducer

