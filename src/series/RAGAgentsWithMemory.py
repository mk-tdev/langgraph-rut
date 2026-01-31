from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langgraph_viz import visualize
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from langchain_core.tools.retriever import create_retriever_tool
except ImportError:  # pragma: no cover
    try:
        from langchain.tools.retriever import create_retriever_tool
    except ImportError:  # pragma: no cover
        try:
            from langchain.tools import create_retriever_tool
        except ImportError:  # pragma: no cover
            from langchain_community.tools import create_retriever_tool

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# Initialize Ollama with the granite-embedding model
embeddings = OllamaEmbeddings(model="granite-embedding:278m")

docs = [
    Document(
        page_content="Artic Vista is owned by Artic Group, a renowned chef with over 20 years of experience in the culinary industry.",
        metadata={"source": "owner.txt"},
    ),
    Document(
        page_content="Artic Vista is located in the heart of the city, offering a unique dining experience with its stunning architecture and ambiance.",
        metadata={"source": "location.txt"},
    ),
    Document(
        page_content="Artic Vista offers a diverse menu featuring both traditional and modern cuisine, with a focus on fresh, locally-sourced ingredients.",
        metadata={"source": "menu.txt"},
    ),
    Document(
        page_content="Artic Vista is known for its exceptional service and attention to detail, ensuring a memorable dining experience for every guest.",
        metadata={"source": "service.txt"},
    ),
    Document(
        page_content="Artic Vista is open from 11:00 AM to 10:00 PM, Monday to Sunday, with special events and private dining options available.",
        metadata={"source": "hours.txt"},
    ),
    Document(
        page_content="Artic Vista is committed to sustainability and environmental responsibility, using eco-friendly practices in all aspects of operations.",
        metadata={"source": "sustainability.txt"},
    ),
    Document(
        page_content="Artic Vista has received numerous awards and accolades for its exceptional cuisine, service, and overall dining experience.",
        metadata={"source": "awards.txt"},
    ),
]

db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

template = """
  Answer the question based on only the following context and chat history. Especially take the latest question.

  Chat History: {chat_history}
  
  Context: {context}
  
  Question: {question}
  """
prompt = ChatPromptTemplate.from_template(template=template)

llm = ChatOllama(model="gpt-oss:120b-cloud")
rag_chain = prompt | llm


class AgentState(TypedDict):
    messages: list[BaseMessage]
    documents: list[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage


class GradeQuestion(BaseModel):
    """
    Boolean value to check whether a question is related to the restaurant Artic Vista
    """

    score: str = Field(
        description="Question is about restrauant? If yes -> 'Yes' if not -> 'No'"
    )


def question_rewrite(state: AgentState):
    print(f"Question rewrite: {state}")

    state["documents"] = []
    state["on_topic"] = ""
    state["rephrased_question"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    if len(state["messages"]) > 1:
        converation = state["messages"][:-1]
        current_question = state["question"].content

        messages = [
            SystemMessage(
                content="You are a helpful assistant that repharases the user's question to be a standalone query."
            )
        ]

        messages.extend(converation)
        messages.append(HumanMessage(content=current_question))
        rephrased_prompt = ChatPromptTemplate.from_messages(messages)
        prompt = rephrased_prompt.format()
        result = llm.invoke(prompt)
        print(f"rephrased_question {result.content.strip()}")
        state["rephrased_question"] = result.content.strip()
    else:
        print("rephrased_question {state['question'].content}")
        state["rephrased_question"] = state["question"].content

    return state


def question_classifier(state: AgentState):
    question = state["messages"][-1].content

    system_message = """
    You are a question classifier. Given a question, determine whether it is related to the one of the following topics:

    1. Information about Artic Vista (the restaurant)
    2. Prices of dishes at Artic Vista (the restaurant)
    3. Opening hours of Artic Vista (the restaurant)

    If the question is about any of these topics, respond with a JSON object containing a 'score' field with value 'Yes'. 
    Otherwise, respond with a JSON object containing a 'score' field with value 'No'.
    
    Example responses:
    - For on-topic: {{"score": "Yes"}}
    - For off-topic: {{"score": "No"}}
  """

    human_message = HumanMessage(
        content=f"User question: {state['rephrased_question']}"
    )

    # Create a prompt that asks for a simple Yes/No response
    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    structured_llm = llm.with_structured_output(GradeQuestion)

    # Chain the prompt with the LLM
    chain = grade_prompt | structured_llm

    # Get the response
    response = chain.invoke({})

    state["on_topic"] = response.score.strip()

    print(f"Question: {question}")
    print(f"Response: {response}")
    print(f"Is on topic: {state['on_topic']}")

    return state


def on_topic_router(state: AgentState):
    on_topic = state.get("on_topic", "").strip().lower()

    if on_topic == "yes":
        return "retrieve_documents"
    else:
        return "off_topic_response"


def retrieve_documents(state: AgentState):
    """
    Retrieve documents from the vector store based on the user's question
    """
    documents = retriever.invoke(state["rephrased_question"])
    state["documents"] = documents
    return state


class GradeDocument(BaseModel):
    """
    Boolean value to check whether a document is relevant to the question
    """

    score: str = Field(
        description="Document is relevant to the question? If yes -> 'Yes' if not -> 'No'"
    )


def retrieval_checker(state: AgentState):
    print(f"Retrieval checker: {state}")

    system_message = """
      You are a document relevance checker. Given a question and a document, determine whether the document is relevant to answering the question.

      If the document is relevant, respond with a JSON object containing a 'score' field with value 'Yes'. 
      Otherwise, respond with a JSON object containing a 'score' field with value 'No'.

      Example responses:
      - For relevant document: {{"score": "Yes"}}
      - For irrelevant document: {{"score": "No"}}
    """
    structured_llm = llm.with_structured_output(GradeDocument)

    rephrased_question = state["rephrased_question"]
    relevant_documents = []

    for doc in state["documents"]:
        human_message = HumanMessage(
            content=f"User Question: {rephrased_question}\nRetrieved Document: {doc.page_content}"
        )
        # Create a prompt that asks for a simple Yes/No response
        grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        # Chain the prompt with the LLM
        chain = grade_prompt | structured_llm

        # Get the response
        response = chain.invoke({})

        print(f"Document: {doc.page_content[:30]}...")
        print(f"Response: {response.score.strip()}")

        if response.score.strip().lower() == "yes":
            relevant_documents.append(doc)

    state["documents"] = relevant_documents
    state["proceed_to_generate"] = len(relevant_documents) > 0
    print(
        f"retrieveal_grader - state['proceed_to_generate']: {state['proceed_to_generate']}"
    )
    return state


def proceed_router(state: AgentState):
    rephrase_count = state.get("rephrase_count", 0)

    if state.get("proceed_to_generate", False):
        return "generate_answer"
    elif rephrase_count >= 2:
        return "cannot_answer"
    else:
        return "refine_question"


def refine_question(state: AgentState):
    print(f"Refine question: {state}")

    rephrase_count = state.get("rephrase_count", 0) + 1

    if rephrase_count >= 2:
        print("Maximum rephrase attempts reached.")
        return state

    question_to_refine = state["rephrased_question"]

    messages = [
        SystemMessage(
            content="You are a helpful assistant that refines the user's question to be more specific and related to Artic Vista restaurant."
        ),
        HumanMessage(
            content=f"The previous question was: {question_to_refine}. Please refine it to be more specific to Artic Vista restaurant."
        ),
    ]

    refine_prompt = ChatPromptTemplate.from_messages(messages)
    prompt = refine_prompt.format()
    result = llm.invoke(prompt)
    print(f"refined_question {result.content.strip()}")
    state["rephrased_question"] = result.content.strip()
    state["rephrase_count"] = rephrase_count + 1

    return state


def generate_answer(state: AgentState):
    print(f"Generate answer: {state}")

    if "messages" not in state or state["messages"] is None:
        raise ValueError("Messages list is missing in the state.")

    history = state["messages"]
    documents = state["documents"]
    rephrased_question = state["rephrased_question"]

    response = rag_chain.invoke(
        {
            "chat_history": history,
            "context": "\n".join([doc.page_content for doc in documents]),
            "question": rephrased_question,
        }
    )

    generation = response.content.strip()
    print(f"Generated answer: {generation}")
    state["messages"].append(AIMessage(content=generation))

    return state


def cannot_answer(state: AgentState):
    print(f"Cannot answer: {state}")

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    response = "I'm sorry, but I cannot answer your question based on the available information about Artic Vista."

    print(f"Generated answer: {response}")
    state["messages"].append(AIMessage(content=response))

    return state


def off_topic_response(state: AgentState):
    print(f"Off topic response: {state}")

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    response = "Your question seems to be unrelated to Artic Vista restaurant. Please ask a question related to Artic Vista."

    print(f"Generated answer: {response}")
    state["messages"].append(AIMessage(content=response))

    return state


checkpointer = MemorySaver()

workflow = StateGraph(AgentState)
workflow.add_node("question_rewrite", question_rewrite)
workflow.add_node("question_classifier", question_classifier)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("retrieval_checker", retrieval_checker)
workflow.add_node("refine_question", refine_question)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("cannot_answer", cannot_answer)
workflow.add_node("off_topic_response", off_topic_response)

workflow.add_edge(START, "question_rewrite")
workflow.add_edge("question_rewrite", "question_classifier")

workflow.add_conditional_edges(
    "question_classifier",
    on_topic_router,
    {
        "retrieve_documents": "retrieve_documents",
        "off_topic_response": "off_topic_response",
    },
)
workflow.add_edge("retrieve_documents", "retrieval_checker")
workflow.add_conditional_edges(
    "retrieval_checker",
    proceed_router,
    {
        "generate_answer": "generate_answer",
        "refine_question": "refine_question",
        "cannot_answer": "cannot_answer",
    },
)
workflow.add_edge("refine_question", "retrieve_documents")
workflow.add_edge("generate_answer", END)
workflow.add_edge("cannot_answer", END)
workflow.add_edge("off_topic_response", END)

graph = workflow.compile(checkpointer=checkpointer)
display(Image(graph.get_graph().draw_mermaid_png()))
config = {"configurable": {"thread_id": "1"}}

# # Off topic question example
input_data = {"question": HumanMessage(content="What is the weather in India today?")}
# result = graph.invoke(input=input_data, config=config)
# print(result)
# print(result["messages"][-1].content)

# # On topic question example
relevant_input = {
    "question": HumanMessage(content="What are the opening hours of Artic Vista?")
}
# relevant_result = graph.invoke(input=relevant_input, config=config)
# print(relevant_result)
# print(relevant_result["messages"][-1].content)

with visualize(graph) as viz:
    viz.invoke(input=input_data, config=config)

    viz.invoke(input=relevant_input, config=config)
