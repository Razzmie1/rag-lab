from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from ollama import Client
from dotenv import load_dotenv
import gradio as gr
import os, subprocess, httpx
import chromadb

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

def answer(message, history):
    files = []
    for msg in history:
        if msg["role"] == "user" and isinstance(msg["content"], tuple):
            files.append(msg["content"][0])
    
    for file in message["files"]:
        files.append(file)

    # TODO: Add persistent Chroma vector store
    # TODO: Handle case where no files are uploaded
    documents = SimpleDirectoryReader(input_files=files).load_data()

    index = VectorStoreIndex.from_documents(
        documents=documents,
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=20)],
    )
    query_engine = index.as_query_engine()
    return str(query_engine.query(message["text"]))

demo = gr.ChatInterface(
    fn=answer,
    type="messages",
    title="Llama Index RAG Chatbot",
    description="Upload any text or pdf files and ask questions about them!",
    examples=[{"text": "What is the main topic of this document?", "files": ["sample1.pdf"]}],
    textbox=gr.MultimodalTextbox(file_types=[".txt", ".pdf"]),
    multimodal=True,
)

demo.launch(debug=False)