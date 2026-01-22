from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langgraph_viz import visualize
from operator import add
from pydantic import BaseModel

class InputState(BaseModel):
  string_value: str
  int_value: int

def modify_state(state: InputState) -> InputState:
  string_value = state.string_value + "_modified"
  int_value = state.int_value + 1
  return InputState(string_value=string_value, int_value=int_value)

graph = StateGraph(InputState)
graph.add_node("branch_a", modify_state)
graph.add_node("branch_b", modify_state)
graph.add_edge(START, "branch_a")
graph.add_edge("branch_a", "branch_b")
graph.add_edge("branch_b", END)

graph.set_entry_point("branch_a")

compiled_graph = graph.compile()

display(Image(compiled_graph.get_graph().draw_mermaid_png()))

result = compiled_graph.invoke(InputState(string_value="test", int_value=1))
print(result)