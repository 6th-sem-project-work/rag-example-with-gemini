import enum
import pydantic
from typing import List, Optional

class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"

class Model(str, enum.Enum):
    gemini_pro = "gemini-pro"
    llama2 = "llama2"
    llama2_uncensored = "llama2-uncensored"

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
    model: Optional[Model]
