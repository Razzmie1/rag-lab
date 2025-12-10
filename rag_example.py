from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from ollama import Client
from dotenv import load_dotenv
from pathlib import Path
import chromadb
import os, subprocess, httpx

# Directory with text documents
INPUT_DIR = Path("data/")
# Query for the documents
QUERY = "Summarize Flow Matching in 5 bullet points."


def is_ollama_running(host="http://localhost:11434"):
    try:
        r = httpx.get(f"{host}/api/tags", timeout=2)
        return r.status_code == 200
    except httpx.RequestError:
        return False

if not is_ollama_running():
    subprocess.Popen(["ollama", "serve"])

load_dotenv()
api_key = os.getenv("OLLAMA_API_KEY")

llm_client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {api_key}"},
)

embed_client = Client(
    host="http://localhost:11434",
)

Settings.llm = Ollama(model="gpt-oss:20b-cloud", request_timeout=120.0, client=llm_client)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", client=embed_client)

db = chromadb.PersistentClient()
chroma_collection = db.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

if chroma_collection.count() == 0:
    input_dir = INPUT_DIR
    documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=20)],
        show_progress=True,
    )
else: 
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )

query_engine = vector_index.as_query_engine(similarity_top_k=10)
response = query_engine.query(QUERY)
print(response)