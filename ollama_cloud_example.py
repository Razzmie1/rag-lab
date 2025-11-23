import os, subprocess, httpx
from dotenv import load_dotenv
from ollama import Client

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
    headers={"Authorization": f"Bearer {api_key}"}
)

embed_client = Client(
    host="http://localhost:11434",
)

messages = [
    {
        "role": "user",
        "content": "Hello",
    },
]

answer = llm_client.chat("gpt-oss:20b-cloud", messages=messages)
print(answer["message"]["content"])

embed_input = embed_client.embed(model="nomic-embed-text", input=["Hello world", "How are you?"])
print(embed_input["embeddings"][0][0])