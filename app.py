from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, VectorStoreQuery
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from ollama import Client
from dotenv import load_dotenv
from pathlib import Path
from functools import partial
import gradio as gr
import os, subprocess, httpx
import chromadb

def is_ollama_running(host="http://localhost:11434"):
    try:
        r = httpx.get(f"{host}/api/tags", timeout=2)
        return r.status_code == 200
    except httpx.RequestError:
        return False

def ingest_file(file_path: Path, vector_store: ChromaVectorStore):
    f_name_filter = MetadataFilter(key="file_name", value=file_path.name)
    f_exists_query = VectorStoreQuery(filters=MetadataFilters(filters=[f_name_filter]))
    result = vector_store.query(f_exists_query)
    if not result.ids:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=20)],
            show_progress=True,
        )

def answer(message: dict, history: list, vector_store: ChromaVectorStore):
    for file in message["files"]:
        ingest_file(Path(file), vector_store)

    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )
    query_engine = vector_index.as_query_engine(
        similarity_top_k=5,
        similarity_cutoff=0.4,
    )
    input_text = message["text"]
    result = query_engine.query(input_text)
    if result.source_nodes:
        response = str(result)
        response += "\n\n\nMy answer is based on the following chunks:"
        sorted_chunks = sorted(result.source_nodes, key=lambda node: node.score, reverse=True)
        for chunk in sorted_chunks:
            response += f"\n\nSimilarity Score: {chunk.score:.2f}:"
            response += f"\nChunk: {chunk.node.get_text()}"
            # sum_prompt = f"Summarize this chunk in 1 sentence:\n{chunk.node.get_text()}"
            # sum_response = Settings.llm.complete(sum_prompt)
            # response += f"\nSummary: {sum_response.text}"
        return response
    else:
        return "Please upload at least one file."

def main():
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

    db_client = chromadb.Client()
    chroma_collection = db_client.get_or_create_collection("my_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    answer_fn = partial(answer, vector_store=vector_store)

    demo = gr.ChatInterface(
        fn=answer_fn,
        type="messages",
        title="Llama Index RAG Chatbot",
        description="Upload any text or pdf files and ask questions about them!",
        textbox=gr.MultimodalTextbox(file_types=[".txt", ".pdf"]),
        multimodal=True,
    )
    demo.launch(debug=False)

if __name__=="__main__":
    main()