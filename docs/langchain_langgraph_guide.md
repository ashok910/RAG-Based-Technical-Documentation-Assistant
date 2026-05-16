# LangChain & LangGraph — Complete Reference

## What is LangChain?

LangChain is a framework for developing applications powered by large language models (LLMs). It enables applications that are context-aware and reason about the provided context to determine how to best answer.

## Core Concepts

### LLMs and Chat Models

LangChain supports many LLM providers. Chat models take a sequence of messages and return a message:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke("What is LangChain?")
print(response.content)
```

### Prompt Templates

PromptTemplates help format inputs to the model:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chain = prompt | model
response = chain.invoke({"question": "What is RAG?"})
```

### Output Parsers

Parse model outputs into structured formats:

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# String output
chain = prompt | model | StrOutputParser()

# JSON output
chain = prompt | model | JsonOutputParser()
```

### Document Loaders

Load documents from various sources:

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
)

# PDF
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Web page
loader = WebBaseLoader("https://docs.langchain.com")
docs = loader.load()
```

### Text Splitters

Split documents into manageable chunks:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "],
)
chunks = splitter.split_documents(docs)
```

### Vector Stores

Store and retrieve embeddings:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
)

# Similarity search
results = vectorstore.similarity_search("my query", k=4)
```

### Retrievers

Retrievers expose a `get_relevant_documents` interface:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
docs = retriever.invoke("my question")
```

---

## LangGraph

LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain.

### Core Concepts

**StateGraph**: The main graph type. Nodes are functions that transform the state.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MyState(TypedDict):
    messages: list
    answer: str

graph = StateGraph(MyState)
```

**Nodes**: Python functions that receive state and return a partial state update:

```python
def my_node(state: MyState) -> MyState:
    # Process state
    return {**state, "answer": "processed"}

graph.add_node("my_node", my_node)
```

**Edges**: Define flow between nodes:

```python
# Simple edge
graph.add_edge("node_a", "node_b")

# Conditional edge
def route(state) -> str:
    if state["condition"]:
        return "node_b"
    return "node_c"

graph.add_conditional_edges(
    "node_a",
    route,
    {"node_b": "node_b", "node_c": "node_c"}
)
```

**Entry and Exit Points**:

```python
graph.set_entry_point("first_node")
graph.add_edge("last_node", END)
```

**Compilation and Execution**:

```python
app = graph.compile()
result = app.invoke({"messages": [], "answer": ""})
```

### Building a RAG Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    retry_count: int

def retrieve(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}

def grade_documents(state: RAGState) -> RAGState:
    # Filter relevant documents
    relevant = [d for d in state["documents"] if is_relevant(d, state["question"])]
    return {**state, "documents": relevant}

def generate(state: RAGState) -> RAGState:
    answer = llm.invoke(format_prompt(state["question"], state["documents"]))
    return {**state, "answer": answer.content}

def should_generate(state: RAGState) -> str:
    if state["documents"]:
        return "generate"
    elif state["retry_count"] < 2:
        return "retry"
    return "fallback"

workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", should_generate, {
    "generate": "generate",
    "retry": "retrieve",
    "fallback": "generate",
})
workflow.add_edge("generate", END)

app = workflow.compile()
```

### State Management

LangGraph uses TypedDict for typed state. Best practices:
- Include all fields a node might read or write
- Use Optional for fields that may not always be set
- Track retry counts to prevent infinite loops

```python
class ComplexState(TypedDict):
    input: str
    documents: List[Document]
    answer: str
    retry_count: int       # Track retries
    route_decision: str    # Routing metadata
    error: Optional[str]   # Error handling
    steps: List[str]       # Audit trail
```

### Checkpointers (Persistence)

LangGraph supports persisting state between invocations:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=checkpointer)

# Use thread_id for conversation continuity
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke({"question": "Hello"}, config=config)
```
