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

# Load configuration from elastic.json
with open('elastic.json') as config_file:
    config = json.load(config_file)
# es = get_es_engine(config)

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

    if elastic_config is None:
        st.error("ElasticSearch Config File should be uploaded")

    else:
        config = json.load(elastic_config)
        # st.json(config)
        st.session_state['es'] = get_es_engine(config)
        st.session_state['llm'] = get_llm_engine(project_id=project_id, api_key=api_key, url=url)
        st.session_state['embed'] = get_emb_engine(project_id=project_id, api_key=api_key, url=url)
        return True

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


if __name__ == "__main__":
    if configurate_rag():
        main()
