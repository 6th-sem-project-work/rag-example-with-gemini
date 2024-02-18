from fastapi import FastAPI
import redis
import json
from classes import Message, MessageRole, Query
import qa

rc = redis.Redis(host="localhost", port=6379, decode_responses=True)
app = FastAPI()


@app.get("/mesg_history/get/{client_id}")
def get_message_history(client_id: str):
    history = rc.lrange(f"{client_id}:history", 0, -1)
    if history is None:
        return {"history": []}
    else:
        history = list(map(json.loads, history))
        return {"history": history}

@app.post("/chat/get")
def get_chat_response(query: Query):
    qaobject = qa.QAResponse()
    response = qaobject.get_response(question=query.question, history=query.history)
    return {"user": query.question, "response": response}

@app.post("/mesg_history/push/")
def push_message_history(mesg: Message):
    rc.rpush(
        f"{mesg.client_id}:history",
        json.dumps({"content": mesg.content, "role": mesg.role}),
    )
