from fastapi import FastAPI
import redis
import json

rc = redis.Redis(host="localhost", port=6379, decode_responses=True)
app = FastAPI()

class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"


class Message(pydantic.BaseModel):
    client_id: str
    role: MessageRole
    message: str

@app.get("/mesg_history/get/{client_id}")
def get_message_history(client_id: str):
    history = rc.lrange(f"{client_id}:history", 0, -1)
    if history is None:
        return {"history": []}
    else:
        history = list(map(json.loads, history))
        return {"history": history}

@app.get("/response")
def get_user_response(query: str):
    qaobject = QAResponse()
    response = qaobject.get_response(query=query)
    return {"user": query, "response": response}

@app.post("/mesg_history/push/")
def push_message_history(mesg: Message):
    rc.rpush(
        f"{mesg.client_id}:history",
        json.dumps({"message": mesg.message, "role": mesg.role}),
    )
