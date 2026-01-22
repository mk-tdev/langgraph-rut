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

result = compiled_graph.invoke({"no_change_value": "test", "string_value": "test", "int_value": 1, "list_value": ["test"]})
print(result)
