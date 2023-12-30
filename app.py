# %%
from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
import tempfile



# %%
# Accessing the OPENAI_API_KEY variable
openaiapikey = os.environ.get('OPENAI_API_KEY')

# Accessing the DEEPINFRA_API_TOKEN variable
deeptoken = os.environ.get('DEEPINFRA_API_TOKEN')

# %%
def load_llm(selected_llm):
    models = {
        "meta-llama/Llama-2-7b-chat-hf": {"temperature": 0.1, "repetition_penalty": 1.2, "max_new_tokens": 256, "top_p": 0.95},
        "mistralai/Mistral-7B-Instruct-v0.1": {"temperature": 0.1, "repetition_penalty": 1.2, "max_new_tokens": 256, "top_p": 0.95},
    }
    return DeepInfra(model_id=f"{selected_llm}", model_kwargs=models[selected_llm])

selected_llm = st.sidebar.selectbox("Select LLM", ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1"], index=0)

