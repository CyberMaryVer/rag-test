import os
import json
import streamlit as st

from es_connector import get_es_engine
from vectorization import search_text_in_elastic
from generation import get_llm_engine, get_emb_engine
from prompts import SYSTEM_PROMPT, TEST_QUERY, TEST_HISTORY, USER_PROMPT

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

INDEX_NAME = 'clalit-ai-poc'

st.set_page_config(page_title=None, page_icon=":home:", layout="wide", initial_sidebar_state="auto", menu_items=None)
st.session_state['rag_config_ready'] = False

def configurate_rag():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.expander("Search settings"):
        st.session_state['chunks_num'] = st.number_input("NUMBER OF DOCUMENTS", min_value=1, max_value=10, value=6)
        st.session_state['index_name'] = st.text_input("INDEX NAME", INDEX_NAME)
        project_id = st.text_input("PROJECT_ID", os.getenv("PROJECT_ID"))
        url = st.text_input("URL", os.getenv("PROJECT_URL"))
        api_key = st.text_input("API_KEY", os.getenv("API_KEY"))
        elastic_config = st.file_uploader("Upload ElasticSearch Config File", type="json")
    st.session_state['llm'] = get_llm_engine(project_id=project_id, api_key=api_key, url=url)
    st.session_state['embed'] = get_emb_engine(project_id=project_id, api_key=api_key, url=url)

    if elastic_config is not None:
        config = json.load(elastic_config)
        st.session_state['es'] = get_es_engine(config)

    if elastic_config is None and 'es' not in st.session_state.keys():
        st.error("ElasticSearch Config File should be uploaded")
        st.session_state['rag_config_ready'] = False

    elif not project_id:
        st.error("PROJECT_ID is required")
        st.session_state['rag_config_ready'] = False

    elif not url:
        st.error("URL is required")
        st.session_state['rag_config_ready'] = False

    elif not api_key:
        st.error("API_KEY is required")
        st.session_state['rag_config_ready'] = False

    elif not st.session_state.get('index_name'):
        st.error("INDEX NAME is required")

    else:
        st.session_state['rag_config_ready'] = True

def main():
    history = st.text_area("HISTORY", TEST_HISTORY)
    user_input = st.text_input("USER PROMPT", TEST_QUERY)

    text_to_search = history + user_input
    generate_llm_response = st.checkbox("Generate Answer with LLM")
    answer_placeholder = st.empty()

    if st.button("Search documents"):
        with st.spinner("Searching in Elastic..."):
            found_documents = search_text_in_elastic(text=text_to_search,
                                                     embedding=st.session_state['embed'],
                                                     search_engine=st.session_state['es'],
                                                     index=st.session_state['index_name'] ,
                                                     search_size=st.session_state['chunks_num'],
                                                     verbose=True)

        for doc in found_documents:
            st.markdown(f"ID: **{doc['doc_id']}** | Score: **{doc['score']}**")
            st.text(doc['passage'])

        st.write(found_documents)

        if generate_llm_response:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(history=history, docs=found_documents)},
                {"role": "user", "content": USER_PROMPT.format(query=user_input)}
            ]

            with st.spinner("Wait for LLM response..."):
                generated_response = st.session_state['llm'].chat(messages=messages)
                content = generated_response['choices'][0]['message']['content']
                answer_placeholder.success(f"LLM Answer: {content}")


def combine_history(limit=5):
    history = ""
    for message in st.session_state.messages[-limit:]:
        history += message['role'].upper() + ": " + message["content"] + "\n"
    return history

@st.fragment
def clear_history():
    st.session_state.messages = []

@st.fragment
def show_history():
    with st.expander("Chat History", expanded=False):
        for message in st.session_state.messages:
            st.write(message)
    return len(st.session_state.messages)

def display_chat():
    col1, col2 = st.columns((2, 1))
    found_documents = []
    input_placeholder = st.empty()

    with col2:
        len_history = show_history()

        if len_history > 0:
            st.warning(f"Pay attention, chat history is not empty, there are {len_history} messages in history")

    with col1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := input_placeholder.chat_input("..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Write user prompt
            with st.chat_message("user"):
                st.markdown(prompt)
            # Write LLM response
            with st.chat_message("assistant"):
                history = combine_history()
                search_query = history + "\n\n" + prompt
                found_documents = search_text_in_elastic(text=search_query,
                                                         embedding=st.session_state['embed'],
                                                         search_engine=st.session_state['es'],
                                                         index=st.session_state['index_name'],
                                                         search_size=st.session_state['chunks_num'],
                                                         verbose=False)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT.format(history=history, docs=found_documents)},
                    {"role": "user", "content": USER_PROMPT.format(query=prompt)}
                ]

                with st.spinner("Wait for LLM response..."):
                    generated_response = st.session_state['llm'].chat(messages=messages)
                    content = generated_response['choices'][0]['message']['content']
                    st.success(f"LLM Answer: {content}")
            st.session_state.messages.append({"role": "assistant", "content": content})

    with col2:
        st.info("First document from search results")
        for idx, doc in enumerate(found_documents[:1]):
            html = f"""
    <p style="margin-bottom: 10px; padding: 10px; border-bottom: 1px solid #ddd;">
        <strong style="background-color: #f0f8ff; color: #0056b3; padding: 4px; border-radius: 4px;">ID: {doc['doc_id']}</strong> | 
        <strong style="background-color: #ffe4b5; color: #d2691e; padding: 4px; border-radius: 4px;">Score: {doc['score']:.2f}</strong>
    </p>
            """
            st.markdown(html, unsafe_allow_html=True)
            st.text(doc['passage'])

        with st.expander("All documents used for generation", expanded=False):
            st.write(found_documents)



if __name__ == "__main__":
    start = st.button("Restart Session")
    if start:
        clear_history()

    if not st.session_state['rag_config_ready']:
        configurate_rag()
    else:
        st.write(st.session_state['rag_config_ready'])
    display_chat()
