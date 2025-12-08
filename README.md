# langgraph rut

### install dependencies
`uv add -r requirements.txt`
##

## What is LangGraph?
LangGraph is a library for building language models.

### State
State is a dictionary that can be passed around between nodes.

### Nodes
Nodes are the basic building blocks of a LangGraph. They are functions that take in some input and return some output.

### Graph
A graph is a collection of nodes that are connected to each other.

### Edges
Edges are the connections between nodes.

### Conditional Edges
Conditional edges are edges that are only taken if a certain condition is met.

### Start Node
Start nodes are the entry points of a graph. They are nodes that do not have any incoming edges.

### End Node
End nodes are the exit points of a graph. They are nodes that do not have any outgoing edges.

### Tools
Tools are nodes that can be called by other nodes. They are used to perform some action.

  - tool nodes are nodes that perform some action.
  - tool edges are edges that connect a tool node to another node.
  - tool calls are the calls to a tool node.

### StateGraph
StateGraph is a class in LangGraph used to build a graph of nodes and edges.

### Runnable
Runnable is a class in LangGraph used to run a graph.

## Messages
Messages are the input and output of a node.

- Human Message: A message from a human.
- AI Message: A message from an AI.
- System Message: A message from the system.
- Tool Message: A message from a tool.
- Function Message: A message from a function.
