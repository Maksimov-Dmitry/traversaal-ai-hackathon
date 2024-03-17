from src import streamlit_utils
from src.prompts import AGENT_SYSTEM_PROMPT, AGENT_USER_PROMPT, RAG_USER_PROMPT, TRAVERSIALAI_USER_PROMPT
from src.rag import RAG

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
import re
import requests
import os
from qdrant_client import QdrantClient

collection_name = 'hotels'

st.set_page_config(page_title="Hotels search chatbot", page_icon="⭐")
st.header('Hotels search chatbot')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/2_%E2%AD%90_context_aware_chatbot.py)')

st.sidebar.header('Choose your preferences')
n_hotels = st.sidebar.number_input('Number of hotels', min_value=1, max_value=10, value=3)


@st.cache_resource
def get_db_client(path='data/db'):
    client = QdrantClient(path=path)
    return client


def add_new_info(chat_history, queries):
    for query in queries:
        chat_history.add_user_message(query)
        chat_history.add_ai_message('Ok, got it!')


def check_params(params):
    changed_params = []

    if 'prev_params' not in st.session_state:
        st.session_state.prev_params = {'city': '<BLANK>', 'price': '<BLANK>', 'rating': '<BLANK>'}

    if st.session_state.prev_params['city'] != params['city']:
        changed_params.append(f'city: I want to find hotels in {params["city"]}' if params['city'] else 'I want to find hotels in any city')

    if st.session_state.prev_params['price'] != params['price']:
        changed_params.append(f'price: I want to find hotels in price range {params["price"]}' if params['price'] else 'I want to find hotels in any price range')

    if st.session_state.prev_params['rating'] != params['rating']:
        changed_params.append(f'rating: I want to find hotels with rating greater than {params["rating"]}')

    st.session_state.prev_params = params

    return changed_params


def get_parameters(db_client):
    points, _ = db_client.scroll(
        collection_name=collection_name,
        limit=1e9,
        with_payload=True,
        with_vectors=False,
    )
    cities = ['Doest not matter'] + list(set([point.payload['city'] for point in points]))
    city = st.sidebar.selectbox('City', list(cities), index=0)
    if city == 'Doest not matter':
        city = None

    prices = ['Doest not matter'] + list(set([point.payload['price'] for point in points]))
    price = st.sidebar.selectbox('Price', list(prices), index=0)
    if price == 'Doest not matter':
        price = None

    rating = st.sidebar.slider('Min hotel rating', min_value=.0, max_value=5.0, value=4.5, step=.5)
    return dict(city=city, price=price, rating=rating)


class HotelsSearchChatbot:
    def __init__(self, db_client):
        streamlit_utils.configure_api_keys()

        self.llm_model = "gpt-4-1106-preview"
        self.temperature = 0.6

        self.embeedings_model = "text-embedding-3-large"
        self.rerank_model = 'rerank-multilingual-v2.0'

        self.ares_api_key = os.environ.get("ARES_API_KEY")
        self.db_client = db_client

    def _traversialai(self, query):
        url = "https://api-ares.traversaal.ai/live/predict"

        payload = {"query": [query]}
        headers = {
            "x-api-key": self.ares_api_key,
            "content-type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        try:
            return response.json()['data']['response_text']
        except:
            return None

    def _get_action(self, text):
        action_pattern = r"Action:\s*(.*)\n"
        action_input_pattern = r"Action Input:\s*(.*)"

        action_match = re.search(action_pattern, text)
        action_input_match = re.search(action_input_pattern, text)

        action = action_match.group(1) if action_match else None
        action_input = action_input_match.group(1) if action_input_match else None
        return action, action_input

    def _make_action(self, action, action_input, rag, chain, chat_history, config, rag_params):
        if action == 'nothing':
            st.markdown(action_input)
            return action_input

        if action == 'hotels_data_base':
            context = rag(action_input, top_k=n_hotels, **rag_params)
            chat_history.add_user_message(RAG_USER_PROMPT.format(context=context, query=action_input))
            response = chain.invoke({"messages": chat_history.messages}, config)
            chat_history.messages.pop()
            return response.content

        if action == 'ares_api':
            context = self._traversialai(action_input)
            chat_history.add_user_message(TRAVERSIALAI_USER_PROMPT.format(context=context, query=action_input))
            response = chain.invoke({"messages": chat_history.messages}, config)
            chat_history.messages.pop()
            return response.content

        return None

    @st.cache_resource
    def setup_chain(_self):
        rag = RAG(embedding_model=_self.embeedings_model, llm_model=_self.llm_model,
                  rerank_model=_self.rerank_model, db_client=_self.db_client, db_collection=collection_name)

        chat_history = ChatMessageHistory()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    AGENT_SYSTEM_PROMPT,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chat = ChatOpenAI(model=_self.llm_model, temperature=_self.temperature, streaming=True)
        chain = prompt | chat

        return chain, chat_history, rag

    @streamlit_utils.enable_chat_history
    def main(self, params):
        chain, chat_history, rag = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            streamlit_utils.display_msg(user_query, 'user')
            queries = check_params(params)
            add_new_info(chat_history, queries)

            chat_history.add_user_message(AGENT_USER_PROMPT.format(input=user_query))
            action_response = chain.invoke({"messages": chat_history.messages})
            chat_history.messages.pop()

            action, action_input = self._get_action(action_response.content)
            with st.chat_message("assistant"):
                st_cb = streamlit_utils.StreamHandler(st.empty())
                response = self._make_action(action, action_input,
                                             rag, chain, chat_history, {"callbacks": [st_cb]}, params)
                chat_history.add_user_message(user_query)
                if response is None:
                    response = 'Sorry, I cannot help you with it. Could you rephrase your question?'
                    st.markdown(response)
                chat_history.add_ai_message(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    db_client = get_db_client()
    params = get_parameters(db_client)
    obj = HotelsSearchChatbot(db_client)
    obj.main(params)