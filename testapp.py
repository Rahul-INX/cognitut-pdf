from langchain.document_loaders import PyPDFDirectoryLoader, TextLoader, CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, EmbeddingsFilter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import DeepInfra
from langchain.chains import ConversationalRetrievalChain
from langchain_core.utils.env import get_from_env
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
import tempfile
import os
from dotenv import load_dotenv
import time
import pandas as pd
import numpy as np
import requests

load_dotenv()

# Accessing the OPENAI_API_KEY variable
openaiapikey = os.environ.get('OPENAI_API_KEY')
deeptoken = os.environ.get('DEEPINFRA_API_TOKEN')

# Accessing the DEEPINFRA_API_TOKEN variable
compressor_llm = DeepInfra(model_id="mistralai/Mistral-7B-Instruct-v0.1")
compressor_llm.model_kwargs = {
    "temperature": 0.1,
    "repetition_penalty": 1.2,
    "max_new_tokens": 500,
    "top_p": 0.90,
}

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# Helper function for printing docs
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

# Cache the document loader function
@st.cache_data( show_spinner=True,persist="disk")
def doc_loader(doc_path):
    # Load documents
    loader = PyPDFDirectoryLoader(doc_path)
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(docs)
    return documents

st.set_page_config(layout="wide", page_title="COGNITUT")
st.title(":red[COGNITUT]")

with st.chat_message("ai", avatar='👨‍🏫'):
    st.write('Hi, how may I help you today ?')

st.sidebar.title("WELCOME TO COGNITUT!")
doc_mode = st.sidebar.toggle(label="Chat With Documents", value=True)
if doc_mode:
    compression_mode=st.sidebar.toggle(label="Advanced Context Compression")




# NOTE: THE KEY OF THE DICTIONARY IS CASE-SENSITIVE
all_subjects = {
    'CSE': {
        'Semester 7': ['Cloud Computing', "Principles Of Management", 'Cyber Forensics', 'Cryptography And Network Security', 'Blockchain Technology'],
        # Add subjects for other semesters as needed
    },
    'IT': {
        # 'Semester 5': ['IT Subject1', 'IT Subject2', 'IT Subject3'],
        # 'Semester 6': ['IT SubjectA', 'IT SubjectB', 'IT SubjectC'],
        # Add subjects for other semesters as needed
    },
    # Add subjects for other departments as needed
}

# Department selection
department = st.sidebar.selectbox(
    'Choose your department (only CSE support yet)',
    ('CSE', 'IT', 'ECE', 'MECH', 'CSBS', 'AI & DS', 'EEE', 'BME'),
    index=0
)

# Semester selection
semester = st.sidebar.selectbox(
    'Choose your current semester (only 7th sem support yet)',
    ('Semester 1', 'Semester 2', 'Semester 3', 'Semester 4', 'Semester 5', 'Semester 6', 'Semester 7', 'Semester 8'),
    index=6
)

# Subject selection based on the selected department and semester
subject_options = all_subjects.get(department, {}).get(semester, [])
subject = st.sidebar.selectbox('Choose your subject', subject_options)


chat_model_name=st.sidebar.selectbox(
    'Choose Chat Model',
    ('meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1'),
    index=0
    )

if department and semester and subject:
    # Path to pdf documents
    doc_path = f"Documents/{department}/{semester}/{subject}"

    # Define the path for the FAISS vector store
    FAISS_DB_PATH = f'vectorStore/{department}/{semester}/{subject}'

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(doc_path):
        os.makedirs(doc_path)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(FAISS_DB_PATH):
        os.makedirs(FAISS_DB_PATH)

    # Check if FAISS_DB_PATH is empty
    if not os.listdir(FAISS_DB_PATH):
        if os.listdir(doc_path):
            # Display loading message while creating the FAISS database
            with st.spinner("Vector DB is creating..."):
                # Load documents using the cached function
                documents = doc_loader(doc_path)

                # Create FAISS database
                db = FAISS.from_documents(documents, embeddings)
                db.save_local(FAISS_DB_PATH)

                st.balloons()
                st.toast(f"Vector DB created in: {FAISS_DB_PATH[12:]}", icon='✅')

        else:
            st.warning("Document directory is empty. No documents found.")
    else:
        # FAISS database is not empty
        db = FAISS.load_local(FAISS_DB_PATH, embeddings)
        st.toast(f"Vector DB loaded from: {FAISS_DB_PATH[12:]}", icon='✅')

        faiss_retriever = db.as_retriever(search_kwargs={"k": 3})

        
        with st.spinner("Documents are loading........"):
            # Load documents using the cached function 
            documents = doc_loader(doc_path)

            st.toast(f"Documents loaded.....", icon='✅')

            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 5

            # initialize the ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever], 
                weights=[0.2, 0.8]
                )





            user_query = st.chat_input('Enter your query here ....')
            docs = ensemble_retriever.get_relevant_documents(user_query)



            if (compression_mode):   ##this must be made into a mode button in the streamlit side bar (retrieval processing -> quality or standard)
                compressor = LLMChainExtractor.from_llm(compressor_llm)
            else:
                compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
                print("using similarity_threshold")
                
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
            compressed_docs = compression_retriever.get_relevant_documents(user_query)

            # st.write(compression_mode)
            # st.write(compressed_docs)

            metadata_info=[]
            for i in range(len(compressed_docs)):    
                metadata_info.append('\''+str(compressed_docs[i].metadata)[23:])

            # st.write(metadata_info)



            def chat_model(model_name):
                llm = DeepInfra(model_id=model_name)
                llm.model_kwargs = {
                    "temperature": 0.2,
                    "repetition_penalty": 1.2,
                    "max_new_tokens": 512,
                    "top_p": 0.9,
                }
                return llm

            llm=chat_model(chat_model_name)

            context = "\n".join([f"{doc.page_content}\nMetadata: {doc.metadata}" for doc in compressed_docs])
            response = llm( context=context,prompt=user_query)

            st.write(response)
            # st.write(metadata_info)