import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  


if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()

if "agent" not in st.session_state:
    st.session_state.agent = create_agent(
        "ollama:llama3",
        system_prompt="You are a helpful assistant. Remember details from earlier messages in the conversation",
        checkpointer=st.session_state.memory,
    )


# result = llm.invoke("what is the capital of india?")

def get_response(input): 
    response = st.session_state.agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": input
            }]
        },
        config = {"configurable": {"thread_id": "1"}},
    )

    print(f"response is {response}")
    return response

input = st.text_input("name", key="name")
clicked = st.button("click me", type="primary")

if clicked:
    resp = get_response(input)

    print(f"resp is {resp}")
    st.write(resp['messages'][-1].content)