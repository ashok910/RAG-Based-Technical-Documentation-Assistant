# ChromaDB — Vector Database Guide

## Overview

Chroma is an open-source embedding database. It makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.

## Installation

```bash
pip install chromadb
```

## Basic Usage

```python
import chromadb

# Create a client (in-memory)
client = chromadb.Client()

# Persistent client
client = chromadb.PersistentClient(path="/path/to/db")

# Create or get a collection
collection = client.create_collection(name="my_docs")
```

## Adding Documents

```python
collection.add(
    documents=["This is document one", "This is document two"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["id1", "id2"]
)
```

## Querying

```python
results = collection.query(
    query_texts=["What is document one about?"],
    n_results=2,
    include=["documents", "distances", "metadatas"]
)
```

## LangChain Integration

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    collection_name="rag_collection",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
)

# Add documents
vectorstore.add_documents(documents)

# Retrieve
docs = vectorstore.similarity_search("my query", k=4)

# Persist (LangChain handles this automatically for PersistentClient)
vectorstore.persist()
```

## Metadata Filtering

```python
results = collection.query(
    query_texts=["query"],
    where={"source": "doc1"},           # Exact match
    where_document={"$contains": "keyword"}  # Text filter
)
```

## Collection Management

```python
# List all collections
client.list_collections()

# Delete collection
client.delete_collection("my_docs")

# Count documents
collection.count()

# Get all documents
collection.get(include=["documents", "metadatas"])
```

## FAISS Comparison

| Feature | ChromaDB | FAISS |
|---------|----------|-------|
| Persistence | Built-in | Manual save/load |
| Filtering | Metadata filters | No native filtering |
| Setup | Simple | Simple |
| Scale | Medium | Very large |
| Query speed | Fast | Fastest |
| Use case | Prototyping, medium datasets | Production, large-scale |

## Best Practices

1. **Choose a chunk size** that balances context (bigger) vs. precision (smaller)
2. **Add rich metadata** (source, page, section) for filtering and citations
3. **Use persistent storage** in production so re-embedding isn't needed on restart
4. **Monitor collection size** — ChromaDB handles millions of vectors well
5. **Normalize embeddings** before inserting for consistent cosine similarity
