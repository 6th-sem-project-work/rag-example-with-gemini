import enum
import pydantic
from typing import List

class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"

class Message(pydantic.BaseModel):
    client_id: str
    role: MessageRole
    content: str

class MessageListItem(pydantic.BaseModel):
    role: MessageRole
    content: str

class Query(pydantic.BaseModel):
    question: str
    history: List[MessageListItem]
