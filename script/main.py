from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import requests

# --- Prompt for Answer Generation --- #
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are Autoaid's assistant. Customers query you to find information based on the document context.
    Follow these rules:
    - Use clean formatting with line breaks
    - Use bullet points where appropriate
    - Use **bold** for keywords
    - Be concise and readable
    - Never show literal '\\n'

    Question: {question}
    Context: {context}
    Answer:
    """
)

# --- Load RAG Components --- #
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("autoaid_faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = Ollama(model="llama3", base_url="http://localhost:11434")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# --- FastAPI Setup ---#
app = FastAPI()

class QueryInput(BaseModel):
    query: str

# --- Helper: Call Ollama directly for self-confidence ---#
def get_model_confidence(answer: str) -> float:
    prompt = f"How confident are you in the following answer on a scale from 0 to 1? Only return the number.\n\nAnswer:\n{answer}"
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    try:
        score = float(response.json()["response"].strip())
        return round(min(max(score, 0.0), 1.0), 2)
    except:
        return 0.5  # fallback

@app.post("/query")
async def query_handler(input: QueryInput):
    result = qa_chain.invoke(input.query)

    # Extract answer and source document scores
    answer = result['result'].replace("\\n", "\n").strip()
    # Ask model to self-evaluate
    model_confidence = get_model_confidence(answer)

    return {
        "answer": answer,
        "model_confidence": model_confidence,
    }
