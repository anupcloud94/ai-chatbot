import uuid
import os

import streamlit as st
from langchain.agents import create_agent
from langchain.agents.middleware import before_agent, after_agent
from langgraph.checkpoint.redis import RedisSaver
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Text,
    Integer,
    ForeignKey,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import logging
import warnings

load_dotenv()  # Load environment variables from .env file

# Suppress all Python warnings at the interpreter level
warnings.filterwarnings("ignore")
# Suppress transformers warnings specifically in Streamlit's logger
logging.getLogger("transformers").setLevel(logging.ERROR)

REDIS_URL = os.getenv("REDIS_URL")

# ======================== db setup ========================
DATABASE_URL = os.getenv("DATABASE_URL")

Db_engine = create_engine(DATABASE_URL)
metadata = MetaData()
Base = declarative_base()

Session = sessionmaker(bind=Db_engine)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    thread_id = Column(String)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    thread_id = Column(String)
    content = Column(Text)

def create_conversation(thread_id):
    session = Session()
    conversation = Conversation(
        thread_id=thread_id
    )
    session.add(conversation)
    session.commit()

def create_message(thread_id, content):
    session = Session()

    result = session.query(Conversation).filter_by(thread_id=thread_id).first()
    message = Message(
        conversation_id=result.id,
        thread_id=thread_id,
        content=content
    )
    session.add(message)
    session.commit()

Base.metadata.create_all(Db_engine)

# ========================= end of db setup =========================

@before_agent
def before_agent_callback(state, runtime):
    create_message(
        thread_id=runtime.execution_info.thread_id,
        content=state["messages"][-1].content,
    )
    return None


@after_agent
def after_agent_callback(state, runtime):
    print(f"After Agent - Thread ID: {runtime.execution_info.thread_id}", flush=True)
    create_message(
        thread_id=runtime.execution_info.thread_id,
        content=state["messages"][-1].content,
    )
    return None


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
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    agent = create_agent(
        llm,
        system_prompt="You are a helpful assistant. Remember details from earlier messages in the conversation",
        checkpointer=checkpointer,
        middleware=[before_agent_callback, after_agent_callback],
    )

    create_conversation(st.session_state.thread_id)  # Create a new conversation in the database for this thread

    return agent



if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = (
        uuid.uuid4().hex
    )  # Generate a unique thread ID for the conversation


def get_response(input):
    print(f"getting response for thread id : {st.session_state.thread_id}")
    response = get_agent().invoke(
        {"messages": [{"role": "user", "content": input}]},
        config={"configurable": {
            "thread_id": st.session_state.thread_id,
        }},
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
            assistant_message = response["messages"][-1].content
            st.markdown(assistant_message)

    # Add assistant message to display
    st.session_state.display_messages.append(
        {"role": "assistant", "content": assistant_message}
    )
