from LLM.GR import GetResponses

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Type your question here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        GR=GetResponses()
        qa_chain=GR.transform()
        result = qa_chain({"username": "user", "question": user_query, "chat_history": st.session_state.chat_history})
        answer = result["answer"]
        answer = answer.replace('\n', '').replace("'", "\\'")
        response = st.write(answer)
    if response != None:
        st.session_state.chat_history.append(AIMessage(content=response))