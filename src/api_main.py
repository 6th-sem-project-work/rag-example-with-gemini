
from fastapi import FastAPI
import redis
import json
from qa import QAResponse

rc = redis.Redis(host='localhost', port=6379, decode_responses=True)
app = FastAPI()

@app.get("/mesg_history/{client_id}")
def get_message_history(client_id: str):
    history = rc.lrange(f"{client_id}:history", 0, -1)
    if history is None:
        return { "history": [] }
    else:
        history = list(map(json.loads, history))
        return { "history": history }

@app.get("/response")
def get_user_response(query: str):
    qaobject = QAResponse()
    response = qaobject.get_response(query=query)
    return {"user": query, "response": response}

