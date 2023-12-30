from langchain.document_loaders import PyPDFDirectoryLoader, TextLoader, CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, EmbeddingsFilter
from langchain.llms import DeepInfra
from langchain.chains import ConversationalRetrievalChain
from langchain_core.utils.env import get_from_env
from dotenv import load_dotenv
import tempfile
import os



# # Set environment variables
# load_dotenv()
# deepinfra_api_token = st.secrets["DEEPINFRA_API_TOKEN"]
# if deepinfra_api_token:
#     os.environ["DEEPINFRA_API_TOKEN"] = deepinfra_api_token
"""uncomment this code when hosting on streamlit"""



# Accessing the OPENAI_API_KEY variable
openaiapikey = os.environ.get('OPENAI_API_KEY')

# Accessing the DEEPINFRA_API_TOKEN variable
deeptoken = os.environ.get('DEEPINFRA_API_TOKEN')


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

loader = PyPDFDirectoryLoader("Documents")
docs = loader.load()
print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap  = 300,
    length_function = len,
    is_separator_regex = False,
)

documents  = text_splitter.split_documents(docs)

%%time
# Define the path for the FAISS vector store
FAISS_DB_PATH = 'vectorstore/db_faiss'

if os.path.exists(FAISS_DB_PATH):
   db = FAISS.load_local(FAISS_DB_PATH, embeddings)
   print("already present")

else:
    FAISS_DB_PATH = 'vectorstore/db_faiss'
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(FAISS_DB_PATH)
    print("vectorDB created")


faiss_retriever = db.as_retriever(search_kwargs={"k": 4})
#" k   define no of top relevant documents to be retrieved"

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 4  #" k   define no of top relevant documents to be retrieved"



# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)



##############################################################################################
user_query =  '''name of the author , hoow have they described the preface of the book'''
##############################################################################################



docs = ensemble_retriever.get_relevant_documents(user_query)



if (input("enter quality mode : high or average")=='high'):   ##this must be made into a mode button in the streamlit side bar
    compressor = LLMChainExtractor.from_llm(compressor_llm)
else:
    compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.80)
    print("using similarity_threshold")
    
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

compressed_docs = compression_retriever.get_relevant_documents(user_query)
# pretty_print_docs(compressed_docs)

llm = DeepInfra(model_id="meta-llama/Llama-2-7b-chat-hf")
llm.model_kwargs = {
    "temperature": 0.0,
    "repetition_penalty": 1.2,
    "max_new_tokens": 512,
    "top_p": 0.9,
}

context = "\n".join([f"{doc.page_content}\nMetadata: {doc.metadata}" for doc in compressed_docs])
response = llm(context=context, prompt=user_query)

metadata_info=[]
for i in range(len(compressed_docs)):    
    metadata_info.append('\''+str(compressed_docs[i].metadata)[23:])


