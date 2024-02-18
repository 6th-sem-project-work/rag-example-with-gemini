import enum
import pydantic

class Message(pydantic.BaseModel):
    client_id: str
    role: MessageRole
    message: str

class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"