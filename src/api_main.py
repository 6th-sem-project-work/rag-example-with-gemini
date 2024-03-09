from fastapi import FastAPI
import redis
import json
from classes import Message, MessageRole, Query, Model
import qa

rc = redis.Redis(host="localhost", port=6379, decode_responses=True)
app = FastAPI()
chain = qa.QaService()


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
    if query.model is None:
        query.model = Model.gemini_pro
    response = chain.get_response(question=query.question, history=query.history, model=query.model)
    return {"question": query.question, "response": response['response'], 'context': response['context']}

@app.post("/mesg_history/push/")
def push_message_history(mesg: Message):
    rc.rpush(
        f"{mesg.client_id}:history",
        json.dumps({"content": mesg.content, "role": mesg.role}),
    )
