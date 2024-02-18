import streamlit as st
import uuid
import json
import requests
from classes import Message, MessageRole

BASE_URL = "http://localhost:8000"

def push_mesg(mesg):
  st.session_state.messages.append(mesg)
  msg = Message(role=mesg["role"], content=mesg["content"], client_id=client_id)
  requests.post(url=f"{BASE_URL}/mesg_history/push/", data=msg.json())
  

if "client_id" not in st.session_state:
  id = str(uuid.uuid1())
  id = "40er7db9-cb5d-10eu-brd5-74977959arrf"
  st.session_state.client_id = id

client_id = st.session_state.client_id

st.title("ChatMed")

if "messages" not in st.session_state:
  history = requests.get(f"{BASE_URL}/mesg_history/get/{client_id}").json()["history"]
  if history is None:
    st.session_state.messages = []
  else:
    st.session_state.messages = history

# For displaying message from history on rerun
for message in st.session_state.messages:
  with st.chat_message(message['role']):
    st.markdown(message['content'])


if prompt := st.chat_input("Message..."):
  with st.chat_message("user"):
    st.markdown(prompt)

  push_mesg({"role": MessageRole.user, "content": prompt})
  

  response = requests.post(url=f"{BASE_URL}/chat/get", data=json.dumps({"question": prompt})).json()["response"]
  with st.chat_message("assistant"):
    st.markdown(response)

  push_mesg({"role": MessageRole.assistant, "content": response})
