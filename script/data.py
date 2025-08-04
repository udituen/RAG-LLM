from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("autoaid_faiss_index", embeddings)

# Initialize Ollama LLM
llm = Ollama(model="llama3")  # Or any other local model

# Setup RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)



