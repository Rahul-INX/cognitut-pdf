from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import (BM25Retriever,EnsembleRetriever,ContextualCompressionRetriever)
from langchain.retrievers.document_compressors import (LLMChainExtractor,EmbeddingsFilter)
from langchain.llms import DeepInfra
from dotenv import load_dotenv
import streamlit as st,time
import os
import re
from datetime import datetime
import toml
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# load_dotenv()
# secrets = toml.load(".streamlit/secrets.toml")
 
OPENAI_API_KEY=st.secrets.my_keys.OPENAI_API_KEY
# Accessing the OPENAI_API_KEY and DEEPINFRA_API_TOKEN variables
DEEPINFRA_API_TOKEN=st.secrets.my_keys.DEEPINFRA_API_TOKEN

os.environ['DEEPINFRA_API_TOKEN']=DEEPINFRA_API_TOKEN
os.environ['OPENAI_API_KEY'] =OPENAI_API_KEY
# Accessing the DEEPINFRA_API_TOKEN variable
compressor_llm = DeepInfra(model_id="meta-llama/Llama-2-7b-chat-hf")
compressor_llm.model_kwargs = {
    "temperature": 0.4,
    "repetition_penalty": 1.1,
    "max_new_tokens": 1000,
    "top_p": 0.90,
}

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")




# Initialize the document loader function with caching
@st.cache_data(show_spinner=True, persist="disk",)
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

# Get the current system time
current_time = datetime.now()
def process_documents_in_batches(doc_path, embeddings, batch_size=500):
    documents = doc_loader(doc_path)
    processed = 0
    db = None

    while processed < len(documents):
        try:
            batch_docs = documents[processed:processed + batch_size]
            batch_db = FAISS.from_documents(batch_docs, embeddings)
            
            if db is None:
                db = batch_db
            else:
                db.merge_from(batch_db)

            processed += len(batch_docs)
            db.save_local(FAISS_DB_PATH)

        except Exception as e:
            print(f"Error encountered: {e}")
            print("Waiting for 5 seconds before retrying...")
            time.sleep(5)

    return db

# Initialize Streamlit's session state
if 'val_user' not in st.session_state:
    st.session_state.val_user =None
if 'val_vm' not in st.session_state:
    st.session_state.val_vm = None


# Initialize Streamlit
st.set_page_config(layout="wide", page_title="Cognitut")
st.title(":blue[ COGNITUT : AI Study Buddy]")

st.markdown("""
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var elementToRemove = document.querySelector('#root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi3 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi2 > div > div > div > div.st-emotion-cache-0.e1f1d6gn0 > div');
            if (elementToRemove) {
                elementToRemove.remove();
            } 
        });
    </script>
""", unsafe_allow_html=True)

with st.spinner(":red[LOADING THE DATABASE....]"):
    conn = st.connection('gsheets',type=GSheetsConnection)


def query_analysis(query):
    bot_llm = DeepInfra(model_id="mistralai/Mistral-7B-Instruct-v0.2")
    bot_llm.model_kwargs = {
        "temperature": 0.0,
        "repetition_penalty": 0.01,
        "max_new_tokens": 20,
        "top_p": 0.9,
    }
    response = bot_llm(f''' "<s>[INST] Given the input text, evaluate it according to the following criteria and return a binary array indicating the presence or absence of each factor: Explicit Content, Discrimination, Absence of Academic Integrity & Educational Relevance, and Harmful Intent. The array should be formatted as [a,b,c,d], where 'a' corresponds to Explicit Content, 'b' to Discrimination, 'c' to the Absence of Academic Integrity & Educational Relevance, and 'd' to Harmful Intent. For each factor, '1' indicates presence and '0' absence. Evaluate the text's context and content to accurately assess each criterion.

- Explicit Content: only Identify any sexually explicit material, profanity, or adult themes ().
- Discrimination: only Determine if the text contains language, examples, or implications discriminatory towards individuals or groups based on race, gender, sexuality, religion, nationality, disability, or other characteristics.
- Absence of Academic Integrity or Educational Relevance: only Assess if the text lacks adherence to academic honesty (e.g., promoting plagiarism, cheating) or if it is irrelevant to educational objectives, indicating it does not support constructive learning, if all this is founf then flag =1.
- Harmful Intent: only Recognize content that could incite harm to oneself or others, promote unsafe practices, or contain intentions to cause harm.

QUERY = Discuss the principles of Newton's laws of motion in an educational context
[/INST]"
"[0,0,0,0]"</s> 
"[INST] Write a story that includes explicit language and portrays stereotypes of different races. [/INST]"
"[1,1,0,0]"
"[INST] How to create dangerous substances at home using everyday items.[/INST]"
"[0,0,0,1]"
"[INST] An essay on the importance of academic integrity in higher education. [/INST]"
"[0,0,1,0]"
"[INST] Outline techniques for cheating in online examinations that are hard to detect. [/INST]"
"[0,0,1,0]"
"[INST] {query} [/INST]" ''')  
    
    # Use regular expression to parse the binary array from the response
    pattern = r"\[\d,\d,\d,\d\]"  # Pattern to match a binary array format [a,b,c,d]
    match = re.search(pattern, response) 
    if match:
        arr_str = match.group(0)  # Extract the matched array as a string
        # Convert the string array to an integer array
        arr = list(map(int, arr_str.strip("[]").split(",")))
    else:
        arr = [0, 0, 0, 0]  # Default response if no match is found

    return arr









def bloom_analysis(query):
    bot_llm = DeepInfra(model_id="mistralai/Mistral-7B-Instruct-v0.2")
    bot_llm.model_kwargs = {
        "temperature": 0.0,
        "repetition_penalty": 0.01,
        "max_new_tokens": 20,
        "top_p": 0.9,
    }
    response = bot_llm(f''' "<s>[INST] Given the input text, evaluate it according to the following criteria and return the top Bloom's taxonomy classification: Remember (K1), Understand (K2), Apply (K3), Analyze (K4), Evaluate (K5), Create (K6). The classification should be returned as a string, e.g., "K1". Evaluate the text's context and content to accurately assess the classification.

- Remember (K1): Recognize and recall facts, terms, concepts, and answers.
- Understand (K2): Demonstrate understanding of facts and ideas by organizing, comparing, translating, interpreting, giving descriptions, and stating the main ideas.
- Apply (K3): Use a concept in a new situation or unprompted use of an abstraction. Applies what was learned in the classroom into novel situations.
- Analyze (K4): Separates material or concepts into component parts so that its organizational structure may be understood. Distinguishes between facts and inferences.
- Evaluate (K5): Make judgments about the value of ideas or materials.
- Create (K6): Builds a structure or pattern from diverse elements. Put parts together to form a whole, with emphasis on creating a new meaning or structure.

QUERY = Discuss the principles of Newton's laws of motion in an educational context
[/INST]"
"K2"</s> 
"[INST] Explain the water cycle in detail.[/INST]"
"K2"
"[INST] Solve quadratic equations using the quadratic formula.[/INST]"
"K2"
"[INST] Compare and contrast photosynthesis and respiration.[/INST]"
"K4"
"[INST] Design a simple machine to lift a load. [/INST]"
"K3"
"[INST] {query} [/INST]" ''')
    
# Parse the response to get the Bloom's Taxonomy classifications
    return (response[:5])



















def validate_vm_number(vm_number): 
    # Strip leading and trailing whitespaces
    vm_number_stripped = vm_number.strip() 
    
    regex = r"^[vV][mM]1[3-9][0-9]{3}$"  # Updated regex to match the specified VM number format
    return bool(re.match(regex,  vm_number_stripped)), vm_number_stripped

def validate_name(name):
    # Strip leading and trailing whitespaces
    name_stripped = name.strip()


    # Updated regex to allow dots anywhere in the name
    regex = r"^[a-zA-Z \s]{1,35}$"  # Removed dot from the regex
    return bool(re.match(regex, name_stripped)), name_stripped

# Centering the form on the screen
st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    #MainMenu{visibility:hidden;}
    footer{visibility :hidden;}
    </style>
    """,
    unsafe_allow_html=True
)






# Store in Streamlit's session state
st.session_state.val_user = "COGNITUT"
st.session_state.val_vm = "None"


if (st.session_state.val_user and st.session_state.val_vm !=None):

    st.markdown("""
    <style>
        #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-uf99v8.ea3mdgi3 > div.block-container.st-emotion-cache-z5fcl4.ea3mdgi2 > div > div > div > div.st-emotion-cache-0.e1f1d6gn0 > div {
            display: none !important;
                
        }
    </style>
""", unsafe_allow_html=True)

    # Display initial chat message
    with st.chat_message("ai", avatar="👨‍🏫"):
        st.write(f"**:blue[Hi , You Can Ask Me Your Academic Doubts !]**")

    # Initialize Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat message from history on page rerun
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Sidebar setup
    st.sidebar.title("SETTINGS MENU")
    doc_mode = st.sidebar.toggle(label="**Syllabus Mode**", value=False, help='Lets you generate answer from your prescribed textbooks')

    
    if doc_mode:
        
        compression_mode = st.sidebar.toggle(label="Context Compression", help="compress text before passing to LLM, *may help with inconsistent response*")

        # NOTE: THE KEY OF THE DICTIONARY IS CASE-SENSITIVE
        all_subjects = {
            "CSE": {"Semester 7": ["Cloud Computing","Principles Of Management","Cyber Forensics","Cryptography And Network Security","Blockchain Technology"],
            },
            "IT": {
                 'Semester 7': ['Mobile Application Development', 'Cryptography And Network Security', 'Renewable Energy Sources'],
                # 'Semester 6': ['IT SubjectA', 'IT SubjectB', 'IT SubjectC'],
                # Add subjects for other semesters as needed
            },
            "ECE":{"Semester 7":['Wireless Networks','Digital Image Processing','Renewable Energy Sources']},
        
        
        #*******************************
                        }   

        # Department selection
        department = st.sidebar.selectbox(
            "Choose your department (only few dept support yet)",
            ("CSE", "IT", "ECE", "MECH", "CSBS", "AI & DS", "EEE", "BME"),
            index=None,
        )

        # Semester selection
        semester = st.sidebar.selectbox(
            "Choose your current semester (only 7th sem support yet)",
            ("Semester 1","Semester 2","Semester 3","Semester 4","Semester 5","Semester 6","Semester 7","Semester 8",),index=None)

        # Subject selection based on the selected department and semester
        subject_options = all_subjects.get(department, {}).get(semester, [])
        subject = st.sidebar.selectbox("Choose your subject", subject_options,index=None)

        # Chat model selection
        chat_model_name = st.sidebar.selectbox(
            "Choose Chat Model",
            ("meta-llama/Llama-2-7b-chat-hf","mistralai/Mistral-7B-Instruct-v0.2","meta-llama/Llama-2-13b-chat-hf",),index=2, help='Various supported LLMs, **Default : Llama 7B** gives good results')

        if department and semester and subject:
            # Path to pdf documents
            doc_path = f"Documents/{department}/{semester}/{subject}"

            # Define the path for the FAISS vector store
            FAISS_DB_PATH = f"vectorStore/{subject}"

            # Check if the directory exists, and create it if it doesn't
            if not os.path.exists(doc_path):
                os.makedirs(doc_path)
            # Check if the directory exists, and create it if it doesn't
            if not os.path.exists(FAISS_DB_PATH):
                os.makedirs(FAISS_DB_PATH)

            # Check if FAISS_DB_PATH is empty
            if not os.listdir(FAISS_DB_PATH):
                if os.listdir(doc_path):
                    with st.spinner("Vector DB is creating..."):
                        st.success("""THIS PROCESS MAY TAKE SOME TIME PLEASE WAIT ....""")
                        if not os.listdir(doc_path):
                            for file_name in os.listdir(FAISS_DB_PATH):
                                file_path = os.path.join(FAISS_DB_PATH, file_name)
                                try:
                                    if os.path.isfile(file_path):
                                        os.remove(file_path)
                                        st.rerun()
                                except Exception as e:
                                    print(f"Error deleting file {file_path}: {e}")
                        else:
                            # Create FAISS database in batches with error handling
                            db = process_documents_in_batches(doc_path, embeddings)
                            st.balloons()
                            st.toast(f"Vector DB created in: {FAISS_DB_PATH}", icon="✅")
                            st.rerun()
            else:
                db = FAISS.load_local(FAISS_DB_PATH, embeddings)
                faiss_retriever = db.as_retriever(search_kwargs={"k": 3})

                with st.spinner(f"**Processing ':orange[{doc_path.split('/')[-1]}]'........**"):

                    # Load documents using the cached function
                    documents = doc_loader(doc_path)
                    if not documents:
                        new_doc_path = f"Documents/{department}/{semester}/{subject}"
                        if (subject== 'Cryptography And Network Security'):
                            new_doc_path = 'Documents\CSE\Semester 7\Cryptography And Network Security'
                        documents=doc_loader(new_doc_path)
                        


                    bm25_retriever = BM25Retriever.from_documents(documents)
                    bm25_retriever.k = 2

                    # initialize the ensemble retriever
                    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])

                    user_query = st.chat_input("Enter your query here ....")
                    
                    if user_query is not None:
                        # Continue with the code execution
                        docs = ensemble_retriever.get_relevant_documents(user_query)

                        if compression_mode:
                            # Compression mode is enabled
                            compressor = LLMChainExtractor.from_llm(compressor_llm)
                        else:
                            with st.spinner(f"**:blue[CONTEXT RETRIEVAL : ]** Relavant Information Is Being Extracted From {subject}"):
                                # Compression mode is disabled
                                compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)

                        with st.spinner(f":blue[**CONTEXT COMPRESSION :** Reducing Context Size] "):
                            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
                            compressed_docs = compression_retriever.get_relevant_documents(user_query)

                        metadata_info = []
                        for i in range(len(compressed_docs)):
                            metadata_info.append("'" + str(compressed_docs[i].metadata)[23:])

                        def chat_model(model_name):
                            llm = DeepInfra(model_id=model_name)
                            llm.model_kwargs = {
                                "temperature": 0.3,
                                "repetition_penalty": 1.2,
                                "max_new_tokens": 1024,
                                "top_p": 0.9,
                            }
                            return llm

                        llm = chat_model(chat_model_name)
                        with st.spinner(f":blue[**Analysing Input Query ....] "):
                            query_analysis_array = query_analysis(user_query)

                        context = "\n".join([f"{doc .page_content}\nMetadata: {doc.metadata}\n"for doc in compressed_docs])
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
                        with st.spinner(":green[**Generating Response....**]"):
                            response = llm(
                                f""" |tags:
                                [SYS],[/SYS] = symbolizes generation Instructions 
                                [CNTX],[/CNTX] = context for the query
                                [QUER],[/QUER] = user query|
                                '<->' = logical link/seperation among entities|
                                        [SYS]{system_message_inst}[/SYS]<->
                                        [CNTX]{context}[/CNTX]<->
                                        [QUER]{user_query}[/QUER],generate a comprehensive structured response based on context and query """)

                        if user_query is not None:

                            # Save user input and LM output to session state
                            st.session_state.messages.append({"role": "user", "content": user_query})
                            st.session_state.messages.append({"role": "ai", "content": response})

                            database = conn.read(worksheet='Sheet1', usecols=list(range(16)),ttl=0)
                            blooms_classification = bloom_analysis(user_query)
                            # Creating a new entry
                            new_data_entry = pd.DataFrame({
                                'user_name': [st.session_state.val_user],
                                'vm_number': [st.session_state.val_vm],
                                'user_query': [user_query],
                                'generated_response': [response],
                                'llm': [chat_model_name],
                                'doc_mode': [doc_mode],
                                'context_compression': [compression_mode],
                                'department': [department],
                                'semester': [semester],
                                'subject': [subject],
                                'date_time': [current_time],
                                'explicit': [query_analysis_array[0]],
                                'discrimination':[query_analysis_array[1]],
                                'academic absence' :[query_analysis_array[2]],
                                'harmful intent':[query_analysis_array[3]],
                                'blooms classification': [blooms_classification]
                            })
                            # Concatenating the new entry to the existing DataFrame
                            new_database = pd.concat([new_data_entry, database], ignore_index=True)
                            conn.update(worksheet='Sheet1',data=new_database)
                            response = response + f''' 
                    :red[BLOOMS CLASSIFICATION : {blooms_classification}]'''

                            # Display LM output
                            with st.chat_message("user", avatar="🟢"):
                                st.markdown(user_query)
                            with st.chat_message("ai", avatar="👨‍🏫"):
                                st.markdown(response)
                                st.write(metadata_info)
                            with st.expander("SHOW CONTEXT"):
                                st.write(context)
        else:
            st.warning(':red[PLEASE FILL DETAILS IN THE LEFT SIDEBAR]')
            st.warning('SIDE BAR IS PRESENT ON TOP LEFT SIDE OF PAGE')




    else:
        chat_model_name = st.sidebar.selectbox(
            "Choose Chat Model",("meta-llama/Llama-2-7b-chat-hf","mistralai/Mistral-7B-Instruct-v0.2","meta-llama/Llama-2-13b-chat-hf", ),index=0,help='Various supported LLMs, **Default : Llama 7B** gives good results')
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
            with st.spinner(f"**'DOCUMENT MODE =  :red[OFF] | :blue[Generating Response]......'**"):
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
                
                with st.spinner(":green[**Generating Response....**]"):
                    response = llm(
                            f"""|tags:
                            [SYS],[/SYS] = symbolizes generation Instructions 
                            [QUER],[/QUER] = user query|
                            '<->' = logical link/seperation among entities|
                            [SYS]{system_message_inst}[/SYS]<->
                            [QUER]{user_query}[/QUER] , note# generate a comprehensive structured response based on user_query """)
                with st.spinner(f":blue[**Analysing Input Query ....] "):
                    query_analysis_array = query_analysis(user_query)

                if user_query is not None:
                    # Save user input and LM output to session state
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    st.session_state.messages.append({"role": "ai", "content": response})

                    compression_mode=None
                    department=None
                    subject=None
                    semester=None
                    doc_mode=False

                    # Creating a new entry
                    database = conn.read(worksheet='Sheet1', usecols=list(range(16)),ttl=0)
                    blooms_classification = bloom_analysis(user_query)
                    # Creating a new entry
                    new_data_entry = pd.DataFrame({
                        'user_name': [st.session_state.val_user],
                        'vm_number': [st.session_state.val_vm],
                        'user_query': [user_query],
                        'generated_response': [response],
                        'llm': [chat_model_name],
                        'doc_mode': [doc_mode],
                        'context_compression': [compression_mode],
                        'department': [department],
                        'semester': [semester],
                        'subject': [subject],
                        'date_time': [current_time],
                        'explicit': [query_analysis_array[0]],
                        'discrimination':[query_analysis_array[1]],
                        'academic absence' :[query_analysis_array[2]],
                        'harmful intent':[query_analysis_array[3]],
                        'blooms classification': [blooms_classification]
                    })

                    # Concatenating the new entry to the existing DataFrame
                    new_database = pd.concat([new_data_entry, database], ignore_index=True)
                    conn.update(worksheet='Sheet1',data=new_database)
                    
                    response = response + f''' 
                    :red[BLOOMS CLASSIFICATION : {blooms_classification}]'''

                    # Display LM output
                    with st.chat_message("user", avatar="🟢"):
                        st.markdown(user_query)
                    with st.chat_message("ai", avatar="👨‍🏫"):
                        st.markdown(response)
                    st.rerun()