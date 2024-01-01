from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import (BM25Retriever,EnsembleRetriever,ContextualCompressionRetriever)
from langchain.retrievers.document_compressors import (LLMChainExtractor,EmbeddingsFilter)
from langchain.llms import DeepInfra
from dotenv import load_dotenv
import streamlit as st
import os
import pandas as pd
import re

load_dotenv()

# Accessing the OPENAI_API_KEY and DEEPINFRA_API_TOKEN variables
openaiapikey = os.environ.get("OPENAI_API_KEY")
deeptoken = os.environ.get("DEEPINFRA_API_TOKEN")

# Accessing the DEEPINFRA_API_TOKEN variable
compressor_llm = DeepInfra(model_id="mistralai/Mistral-7B-Instruct-v0.1")
compressor_llm.model_kwargs = {
    "temperature": 0.4,
    "repetition_penalty": 1,
    "max_new_tokens": 1000,
    "top_p": 0.90,
}

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Check if environment variables are set
if openaiapikey is None or deeptoken is None:
    st.error("Environment variables not set. Please configure OPENAI_API_KEY and DEEPINFRA_API_TOKEN.")
    st.stop()

# Initialize the document loader function with caching
@st.cache_data(show_spinner=True, persist="disk")
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

# Initialize Streamlit
st.set_page_config(layout="wide", page_title="COGNITUT")
st.title(":red[COGNITUT]")

# Display initial chat message
with st.chat_message("ai", avatar="👨‍🏫"):
    st.write("Hi, how may I help you today ?")

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat message from history on page rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Sidebar setup
st.sidebar.title("WELCOME TO COGNITUT!")
doc_mode = st.sidebar.toggle(label="Chat With Documents", value=True)


if doc_mode:
    compression_mode = st.sidebar.toggle(label="Advanced Context Compression")

    # NOTE: THE KEY OF THE DICTIONARY IS CASE-SENSITIVE
    all_subjects = {
        "CSE": {"Semester 7": ["Cloud Computing","Principles Of Management","Cyber Forensics","Cryptography And Network Security","Blockchain Technology",],
        },
        "IT": {
            # 'Semester 5': ['IT Subject1', 'IT Subject2', 'IT Subject3'],
            # 'Semester 6': ['IT SubjectA', 'IT SubjectB', 'IT SubjectC'],
            # Add subjects for other semesters as needed
        },
    }

    # Department selection
    department = st.sidebar.selectbox(
        "Choose your department (only CSE support yet)",
        ("CSE", "IT", "ECE", "MECH", "CSBS", "AI & DS", "EEE", "BME"),
        index=None,
    )

    # Semester selection
    semester = st.sidebar.selectbox(
        "Choose your current semester (only 7th sem support yet)",
        ("Semester 1","Semester 2","Semester 3","Semester 4","Semester 5","Semester 6","Semester 7","Semester 8",),index=None)

    # Subject selection based on the selected department and semester
    subject_options = all_subjects.get(department, {}).get(semester, [])
    subject = st.sidebar.selectbox("Choose your subject", subject_options)

    # Chat model selection
    chat_model_name = st.sidebar.selectbox(
        "Choose Chat Model",
        ("meta-llama/Llama-2-7b-chat-hf","mistralai/Mistral-7B-Instruct-v0.1","meta-llama/Llama-2-13b-chat-hf",),index=0)

    if department and semester and subject:
        # Path to pdf documents
        doc_path = f"Documents/{department}/{semester}/{subject}"

        # Define the path for the FAISS vector store
        FAISS_DB_PATH = f"vectorStore/{department}/{semester}/{subject}"

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
                    st.toast(f"Vector DB created in: {FAISS_DB_PATH[12:]}", icon="✅")

            else:
                st.warning("Document directory is empty. No documents found.")
                st.toast(f"WARNING : Empty Directory\nLoad Files In {doc_path}", icon="⚠")
        else:
            # FAISS database is not empty
            db = FAISS.load_local(FAISS_DB_PATH, embeddings)
            faiss_retriever = db.as_retriever(search_kwargs={"k": 5})

            with st.spinner(f"**Processing ':orange[{doc_path.split('/')[-1]}]'........**"):

                # Load documents using the cached function
                documents = doc_loader(doc_path)

                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 3

                # initialize the ensemble retriever
                ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.2, 0.8])

                user_query = st.chat_input("Enter your query here ....")

                if user_query is not None:
                    # Continue with the code execution
                    docs = ensemble_retriever.get_relevant_documents(user_query)

                    if compression_mode:
                        # Compression mode is enabled
                        compressor = LLMChainExtractor.from_llm(compressor_llm)
                    else:
                        # Compression mode is disabled
                        compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)

                    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
                    compressed_docs = compression_retriever.get_relevant_documents(user_query)

                    metadata_info = []
                    for i in range(len(compressed_docs)):
                        metadata_info.append("'" + str(compressed_docs[i].metadata)[23:])

                    def chat_model(model_name):
                        llm = DeepInfra(model_id=model_name)
                        llm.model_kwargs = {
                            "temperature": 0.1,
                            "repetition_penalty": 1.2,
                            "max_new_tokens": 1024,
                            "top_p": 0.9,
                        }
                        return llm

                    llm = chat_model(chat_model_name)

                    context = "\n".join([f"{doc.page_content}\nMetadata: {doc.metadata}\n"for doc in compressed_docs])
                    system_message_inst = f"""### Role: Specialized Academic Expert Bot
**Description:**
Specialized Academic Expert dedicated to in-depth knowledge in the {subject} domain, delivering precise and structured information. Committed to addressing user queries with clarity and aligning responses with academic principles.
### Task:
1. **Subject-specific Contextual Mastery:**
- Master the art of extracting relevant information in the {subject} context.
- Base responses meticulously on the provided {subject} context.
- Use the relevant information to give a comprehensive answer, adhering strictly to academic standards.
2. **Thorough {subject} Responsiveness:**
- Demonstrate a commitment to addressing user queries comprehensively in the {subject} field.
- Utilize {subject}-specific context extensively to provide detailed responses.
- Stay within the defined context range and avoid making assumptions beyond the provided information.
3. **Art of {subject} Language:**
- Employ clear and concise language tailored for the {subject} academic audience.
- Prioritize clarity through strategic use of headings, markdown, subheadings, paragraphs, and bullet points.
- Utilize markdown techniques such as '#' for titles, '*' for highlighting, and '**' for bolding.
4. **Handling Irrelevance in {subject} Context:**
- Address situations where the {subject} context lacks relevance by responding with "NOT ENOUGH INFORMATION COULD BE FOUND IN THE {subject} CONTEXT..."
- Avoid providing information beyond the specified context range.
5. **Architectural Clarity in {subject} Responses:**
- Craft responses with a robust structure, specific to the {subject} field.
- Ensure that responses are accessible and understandable to individuals without prior {subject} knowledge.
6. **Guiding Academic Principles in {subject} Expertise:**
- Uphold academic principles in every response, maintaining a high standard of accuracy and reliability in the {subject} domain.
- Do not hallucinate on information; stay within the confines of the provided context.
##** Necessity**:
- give a comprehensive, structured answer to the Query
- In the presence of explicit language, comments, vulgar slang, or harmful information, respond with 'I Am a responsible AI. Hence, cannot help you with that' and conclude the response.
"""
                    response = llm(
                        f""" |tags:
                        [INST],[/INST] = symbolizes generation Instructions 
                        [CNTX],[/CNTX] = context for the query
                        [QUER],[/QUER] = user query|
                        '<->' = logical link/seperation among entities|
                                [INST]{system_message_inst}[/INST]<->
                                [CNTX]{context}[/CNTX]<->
                                [QUER]{user_query}[/QUER]
                                """)

                    if user_query is not None:
                        # Save user input and LM output to session state
                        st.session_state.messages.append({"role": "user", "content": user_query})
                        st.session_state.messages.append({"role": "ai", "content": response})

                        # Display LM output
                        with st.chat_message("user", avatar="🟢"):
                            st.markdown(user_query)
                        with st.chat_message("ai", avatar="👨‍🏫"):
                            st.markdown(response)
                            st.write(metadata_info)
                        with st.expander("SHOW CONTEXT"):
                            st.write(context)
    




else:
    chat_model_name = st.sidebar.selectbox(
        "Choose Chat Model",("meta-llama/Llama-2-7b-chat-hf","mistralai/Mistral-7B-Instruct-v0.1","meta-llama/Llama-2-13b-chat-hf", ),index=0)
    user_query = st.chat_input("Enter your query here ....")

    def chat_model(model_name):
        llm = DeepInfra(model_id=model_name)
        llm.model_kwargs = {
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 1024,
            "top_p": 0.85,
        }
        return llm

    llm = chat_model(chat_model_name)
    if user_query is not None:
        with st.spinner(f"**'DOCUMENT MODE =  :red[OFF] | :green[Generating Response]......'**"):
            system_message_inst = """# Role: Academic Expert
**Description:**
Dedicated Comprehensive Academic Expert Bot with a profound understanding of various subjects, committed to delivering high-quality, detailed responses. This bot excels in providing precise and structured information, aligning answers with the highest academic standards.
## Task:
1. **Subject-specific Contextual Mastery:**
- Master the art of extracting relevant information in any academic context.
- Base responses meticulously on the provided subject context.
- Utilize relevant information to deliver comprehensive answers adhering to academic standards.
2. **Thorough Responsiveness:**
- Demonstrate a commitment to addressing user queries comprehensively across diverse academic fields.
- Utilize subject-specific context extensively to provide detailed, well-informed responses.
3. **Art of Academic Language:**
- Employ clear and concise language tailored for a diverse academic audience.
- Prioritize clarity through strategic use of headings, markdown, subheadings, paragraphs, and bullet points.
- Use markdown techniques for titles (#), highlighting (*), and bolding (**).
4. **Handling Irrelevance in Academic Context:**
- Address situations where the academic context lacks relevance by responding with "NOT ENOUGH INFORMATION COULD BE FOUND IN THE CONTEXT..."
5. **Architectural Clarity in Responses:**
- Craft responses with a robust structure applicable to various academic fields.
- Ensure that responses are accessible and understandable to individuals without prior knowledge in a specific subject.
6. **Guiding Academic Principles:**
- Uphold academic principles in every response, maintaining a high standard of accuracy, reliability, and depth across disciplines.
## Markdown Techniques Explanation:
- Use '#' for titles, e.g., #Title#
- For highlighting and bolding, use '*', e.g., *Highlighted* or **Bolded**.
##** Necessity**:
- give a fulfilling , comprehensive, structure answer to the query
- In the presence of explicit language, comments, vulgar slang, or harmful information, respond with 'I Am a responsible AI. Hence, cannot help you with that' and conclude the response.
"""
            
            response = llm(
                        f"""|tags:
                        [INST],[/INST] = symbolizes generation Instructions 
                        [QUER],[/QUER] = user query|
                        '<->' = logical link/seperation among entities|
                        [INST]{system_message_inst}[/INST]<->
                        [QUER]{user_query}[/QUER]""")





            if user_query is not None:
                # Save user input and LM output to session state
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.session_state.messages.append({"role": "ai", "content": response})
                # Display LM output
                with st.chat_message("user", avatar="🟢"):
                    st.markdown(user_query)
                with st.chat_message("ai", avatar="👨‍🏫"):
                    st.markdown(response)
                st.rerun()
