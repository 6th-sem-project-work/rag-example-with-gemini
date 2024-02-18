import streamlit as st
from qa import get_response
import uuid
import redis
import json
import requests
from classes import Message, MessageRole

BASE_URL = "http://localhost:8000"

@st.cache_resource
def redis_client():
  return redis.Redis(host='localhost', port=6379, decode_responses=True)

def push_mesg(mesg):
  st.session_state.messages.append(mesg)
  # rc.rpush(f"{client_id}:history", json.dumps(mesg))
  # r = MessageRole.mesg.role if mesg.role.lower() == "assistant" else "user" 
  msg = Message(role=mesg.role.value, message=mesg.content, client_id=client_id)
  print(msg.json())

  requests.post(url=f"{BASE_URL}/msg_history/push/", data=msg.json())
  

rc = redis_client()

if "client_id" not in st.session_state:
  # id = str(uuid.uuid1())
  id = "40e67db9-cb5d-11ee-bcd5-74977959a8ff"
  st.session_state.client_id = id

client_id = st.session_state.client_id

st.title("ChatMed")

if "messages" not in st.session_state:
  history = rc.lrange(f"{client_id}:history", 0, -1)
  if history is None:
    st.session_state.messages = []
  else:
    history = list(map(json.loads, history))
    st.session_state.messages = history

# For displaying message from history on rerun
for message in st.session_state.messages:
  with st.chat_message(message['role']):
    st.markdown(message['content'])


if prompt := st.chat_input("Message..."):
  with st.chat_message("user"):
    st.markdown(prompt)

  push_mesg({"role": MessageRole.user, "content": prompt})
  

  # response = get_response(prompt)
  response = requests.get(url=f"{BASE_URL}/response", data=json.dumps(prompt))
  with st.chat_message("assistant"):
    st.markdown(response)

  push_mesg({"role": MessageRole.assistant, "content": response})
  # mesg = Message(role=MessageRole.assistant, message=response, client_id=client_id)

  # requests.post(url=f"{BASE_URL}/mesg_history/push/", data=mesg.json())