# Pydantic v2 — Data Validation Guide

## What is Pydantic?

Pydantic is the most widely used data validation library for Python. It uses Python type annotations to define data schemas and validates data at runtime.

## Basic Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class User(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    tags: List[str] = []

# Valid data
user = User(id=1, name="Alice", email="alice@example.com")
print(user.model_dump())
# {'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'age': None, 'tags': []}
```

## Field Validation

```python
from pydantic import BaseModel, Field, field_validator

class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0, description="Price must be positive")
    quantity: int = Field(default=0, ge=0)

    @field_validator('name')
    @classmethod
    def name_must_not_contain_spaces_at_start(cls, v):
        return v.strip()
```

## Model Configuration

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    database_url: str = "sqlite:///./test.db"
    debug: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

settings = Settings()  # Reads from environment variables / .env
```

## Serialization

```python
user = User(id=1, name="Alice", email="alice@example.com")

# To dict
d = user.model_dump()

# To JSON
j = user.model_dump_json()

# From dict
user2 = User.model_validate({"id": 1, "name": "Bob", "email": "bob@example.com"})

# From JSON
user3 = User.model_validate_json('{"id":2,"name":"Carol","email":"carol@example.com"}')
```

## Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str = "US"

class Person(BaseModel):
    name: str
    address: Address
    friends: List['Person'] = []

person = Person(
    name="Alice",
    address={"street": "123 Main St", "city": "Springfield"}
)
```

## Custom Types & Validators

```python
from pydantic import BaseModel, field_validator, model_validator
from typing import Any

class Order(BaseModel):
    item_count: int
    item_price: float
    total: float = 0.0

    @model_validator(mode='after')
    def compute_total(self) -> 'Order':
        self.total = self.item_count * self.item_price
        return self
```

---

# Python Type Hints Quick Reference

## Basic Types

```python
x: int = 5
y: float = 3.14
z: str = "hello"
b: bool = True
```

## Collections

```python
from typing import List, Dict, Tuple, Set, Optional, Union

items: List[str] = ["a", "b", "c"]
mapping: Dict[str, int] = {"a": 1}
pair: Tuple[int, str] = (1, "hello")
unique: Set[int] = {1, 2, 3}
maybe: Optional[str] = None          # Same as Union[str, None]
either: Union[int, str] = 42
```

## Python 3.10+ Syntax

```python
# Union with |
def process(x: int | str) -> None: ...

# Optional shorthand
def find(id: int) -> str | None: ...
```

## TypedDict

```python
from typing import TypedDict, Required, NotRequired

class Movie(TypedDict):
    title: str           # Required
    year: int            # Required
    rating: NotRequired[float]  # Optional key
```

## Generics

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()
```

## Protocol (Structural Subtyping)

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

def render(shape: Drawable) -> None:
    shape.draw()

render(Circle())  # Works without explicit inheritance
```
