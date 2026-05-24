import uuid

import streamlit as st
from langchain.agents import create_agent
from langgraph.checkpoint.redis import RedisSaver


import logging
import warnings

# Suppress all Python warnings at the interpreter level
warnings.filterwarnings("ignore")
# Suppress transformers warnings specifically in Streamlit's logger
logging.getLogger("transformers").setLevel(logging.ERROR)

REDIS_URL = "redis://localhost:6379/0"

# cache the checkpointer and agent to avoid re-creating them on every interaction
@st.cache_resource
def get_checkpointer():
    checkpointer_context = RedisSaver.from_conn_string(REDIS_URL)
    checkpointer = checkpointer_context.__enter__()

    checkpointer.setup()  # Clear previous checkpoints for a clean slate
    return checkpointer

@st.cache_resource
def get_agent():
    checkpointer = get_checkpointer()
    agent = create_agent(
        "ollama:llama3",
        system_prompt="You are a helpful assistant. Remember details from earlier messages in the conversation",
        checkpointer=checkpointer,
    )

    return agent

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex  # Generate a unique thread ID for the conversation


def get_response(input): 
    print(f"getting response for thread id : {st.session_state.thread_id}")
    response = get_agent().invoke(
        {
            "messages": [{
                "role": "user",
                "content": input
            }]
        },
        config = {"configurable": {"thread_id": st.session_state.thread_id}},
    )

    return response

# display previous messages in the conversation
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# when user inputs a new message
if prompt := st.chat_input("Say something..."):

    # add to the list of messages to display
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    
    # add user message to the ui
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
            assistant_message = response['messages'][-1].content
            st.markdown(assistant_message)
    
    # Add assistant message to display
    st.session_state.display_messages.append(
        {"role": "assistant", "content": assistant_message}
    )