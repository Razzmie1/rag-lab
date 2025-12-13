# RAG Lab

## Description

A project to experiment around with RAG on provided text, audio or video

## Installation on Windows

### Setup Repo

Clone this repo to your folder
```shell
git clone https://github.com/Razzmie1/rag-lab.git path/to/your/folder
cd path/to/your/folder
```
Create a `.env` file and define your API key (replace `YOUR_KEY` with your actual Ollama API key)
```shell
echo OLLAMA_API_KEY=YOUR_KEY > .env
```

### Setup Virtual Environment

Create a virtual environment with [anaconda](https://www.anaconda.com/download/success#download)
```shell
conda env create -f environment.yml
conda activate rag_lab
```
Alternatively, you can also create the environment using venv. Note that `python=3.10` was used for this project.
```shell
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Install FFmpeg

To use RAG on an audio file you need to install [FFmpeg](https://www.gyan.dev/ffmpeg/builds/)
```
winget install ffmpeg
```


## Usage

You can run each python scripts with `python script_name.py`

The [ollama_cloud_example](./ollama_cloud_example.py) simply tests if the local embedding and the interaction with the Ollama cloud LLM works.

The [rag_example](./rag_example.py) tests a RAG query on a folder of documents. Here, you should modify the `INPUT_DIR` and `QUERY` to your needs.

The [whisper_example](./whisper_example.py) tests if the transcription of an audio file works. Here, you should modify the `AUDIO_PATH` to your needs.

The actual [app](./app.py) launches a gradio app on a local web server, where you can input a document or audio file and a query for that file. The chatbot will then respond to it using RAG.

## Roadmap

- provide audio timestamps of retrieved nodes as source references
- wrap ollama_cloud_- , rag_-, and whisper_example into test scripts using pytest, to test if setup works
- include option to turn off QA and only do semantic search
- perhaps embed the audio itself and not only the speech transcription, e.g. to search for specific sounds in audio
- same steps for video inputs, extract audio and frames, each with its own embedding

## Features

Ollama, LlamaIndex, Gradio, ChromaDB