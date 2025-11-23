import os
from dotenv import load_dotenv
from ollama import Client

load_dotenv()
api_key = os.getenv("OLLAMA_API_KEY")

client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {api_key}"}
)

messages = [
    {
        "role": "user",
        "content": "Hello",
    },
]

for part in client.chat("gpt-oss:20b-cloud", messages=messages, stream=True):
    print(part["message"]["content"], end="", flush=True)