import streamlit as st
from qa import get_response
import uuid
import redis
import json

@st.cache_resource
def redis_client():
  return redis.Redis(host='localhost', port=6379, decode_responses=True)

def push_mesg(mesg):
  st.session_state.messages.append(mesg)
  rc.rpush(f"{client_id}:history", json.dumps(mesg))

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

  push_mesg({"role": "user", "content": prompt})

  response = get_response(prompt)

  with st.chat_message("assistant"):
    st.markdown(response)

  push_mesg({"role": "assistant", "content": response})
