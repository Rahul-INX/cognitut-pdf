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
import os
import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrieval import BM25Retriever, FAISS, EnsembleRetriever
from langchain.compression import LLMChainExtractor, EmbeddingsFilter, ContextualCompressionRetriever
from langchain.llm import DeepInfra
from langchain.memory import StreamlitChatMessageHistory



load_dotenv()


embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# Accessing the OPENAI_API_KEY variable
openaiapikey = os.environ.get('OPENAI_API_KEY')

deeptoken = os.environ.get('DEEPINFRA_API_TOKEN')



# Set up the Streamlit interface
st.set_page_config(layout="wide", page_title="Cognitut")
st.title(":red[COGNITUT]")

# Set up the chat interface
with st.chat_message("ai", avatar='👨‍🏫'):
    st.write('Hi, how may I help you today?')

# User input field
user_input = st.text_input("User Query")

# Submit button
if st.button("Submit"):
    # Initialize StreamlitChatMessageHistory memory
    chat_memory = StreamlitChatMessageHistory()

    # Function to process user query
    def process_user_query(query):
        user_query = query

        docs = ensemble_retriever.get_relevant_documents(user_query)

        if st.sidebar.radio("Retrieval Processing Mode", options=["High Quality", "Standard"]) == 'High Quality':
            compressor = LLMChainExtractor.from_llm(compressor_llm)
        else:
            compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.80)
            st.info("Using similarity_threshold")

        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
        compressed_docs = compression_retriever.get_relevant_documents(user_query)

        st.write(compressed_docs)

        metadata_info = [str(compressed_docs[i].metadata)[23:] for i in range(len(compressed_docs))]

    # Function to generate response
    def generate_response(query):
        llm = DeepInfra(model_id="meta-llama/Llama-2-7b-chat-hf", deepinfra_api_token=DEEPINFRA_API_TOKEN)
        llm.model_kwargs = {
            "temperature": 0.0,
            "repetition_penalty": 1.2,
            "max_new_tokens": 512,
            "top_p": 0.9,
        }

        # Retrieve chat history from memory
        chat_history = chat_memory.messages

        # Add user query to chat history
        chat_memory.add_user_message(query)

        # Generate response using chat history as context
        context = "\n".join([f"{msg.content}\nMetadata: {msg.metadata}" for msg in chat_history])
        response = llm(context=context, prompt=query)

        # Add AI response to chat history
        chat_memory.add_ai_message(response)

        return response

    process_user_query(user_input)
    response = generate_response(user_input)
    st.chat_message("ai", response)

# NOTE THE KEY OF DICTIONARY ARE CASE SENSITIVE
all_subjects = {
    'CSE': {
        'Semester 7': ['Cloud Computing',"Principles Of Management",'Cyber Forensics','Cryptography And Network Security','Blockchain Technology'],
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
    index=0
)

# Subject selection based on the selected department and semester
subject_options = all_subjects.get(department, {}).get(semester, [])
subject = st.sidebar.selectbox('Choose your subject', subject_options)

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

                # Create FAISS database
                db = FAISS.from_documents(documents, embeddings)
                db.save_local(FAISS_DB_PATH)

                st.balloons()
                st.toast(f"Vector DB created in: {FAISS_DB_PATH[12:]}",icon='✅')
                
        else:
            st.warning("Document directory is empty. No documents found.")
    else:
        # FAISS database is not empty
        db = FAISS.load_local(FAISS_DB_PATH, embeddings)
        st.toast(f"Vector DB loaded from: {FAISS_DB_PATH[12:]}",icon='✅')

    faiss_retriever = db.as_retriever(search_kwargs={"k": 4})
    " k   define no of top relevant documents to be retrieved"

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4
    
    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.2, 0.8]
    )