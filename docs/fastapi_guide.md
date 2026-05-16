# FastAPI — Complete Guide

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.

## Key Features

- **Fast**: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic).
- **Fast to code**: Increase the speed to develop features by about 200% to 300%.
- **Fewer bugs**: Reduce about 40% of human (developer) induced errors.
- **Intuitive**: Great editor support. Completion everywhere. Less time debugging.
- **Easy**: Designed to be easy to use and learn.
- **Short**: Minimize code duplication.
- **Robust**: Get production-ready code with automatic interactive documentation.
- **Standards-based**: Based on (and fully compatible with) the open standards for APIs: OpenAPI and JSON Schema.

## Installation

```bash
pip install fastapi uvicorn[standard]
```

## First Steps

Create a file `main.py` with:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Run the server:

```bash
uvicorn main:app --reload
```

Navigate to `http://127.0.0.1:8000` to see the running application.

## Path Parameters

You can declare path "parameters" or "variables" with the same syntax used by Python format strings:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

FastAPI will validate the type of `item_id` automatically. If you send a non-integer, it returns a clear HTTP error.

## Query Parameters

When you declare other function parameters that are not part of the path parameters, they are interpreted as "query" parameters:

```python
fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```

The query is the set of key-value pairs that go after the `?` in a URL, separated by `&` characters:
- `http://127.0.0.1:8000/items/?skip=0&limit=10`

## Request Body

When you need to send data from a client to your API, you send it as a **request body**. Use Pydantic models:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

## Dependency Injection

FastAPI has a powerful but simple **Dependency Injection** system. It allows you to declare components ("dependencies") that your endpoints need:

```python
from fastapi import Depends

async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

Dependencies can also be classes:

```python
class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(commons: CommonQueryParams = Depends()):
    return commons
```

## Background Tasks

You can define background tasks to be run after returning a response:

```python
from fastapi import BackgroundTasks

def write_notification(email: str, message=""):
    with open("log.txt", mode="w") as email_file:
        content = f"notification for {email}: {message}"
        email_file.write(content)

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}
```

## Middleware

You can add middleware to FastAPI applications. A "middleware" is a function that works with every request before it is processed by any specific path operation, and also with every response before returning it:

```python
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## CORS (Cross-Origin Resource Sharing)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## HTTP Status Codes

- `200` OK: Success
- `201` Created: Resource created
- `204` No Content: Success with no body
- `400` Bad Request: Client error
- `401` Unauthorized: Authentication required
- `403` Forbidden: Authenticated but no permission
- `404` Not Found: Resource not found
- `422` Unprocessable Entity: Validation error
- `500` Internal Server Error: Server-side error
