import enum
import pydantic

class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"

class Message(pydantic.BaseModel):
    client_id: str
    role: MessageRole
    content: str

class Query(pydantic.BaseModel):
    question: str
