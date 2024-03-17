import os
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):

        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute


def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)


def configure_api_keys():
    KEYS = ['OPENAI_API_KEY', 'CO_API_KEY', 'ARES_API_KEY']
    st.sidebar.header('Api Keys Configuration')
    st.markdown(
        """
    <style>
        [title="Show password text"] {
            display: none;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    for key in KEYS:
        if key in os.environ:
            st.session_state[key] = os.environ[key]
        api_key = st.sidebar.text_input(
            label=key,
            type="password",
            value=st.session_state[key] if key in st.session_state else '',
            placeholder="..."
        )
        if api_key:
            st.session_state[key] = api_key
            os.environ[key] = api_key
        else:
            st.error(f"Please add your {key} to continue.")
            st.stop()
