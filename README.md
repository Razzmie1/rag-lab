# RAG Lab

## Description

A project to experiment around with RAG on provided text, audio or video

## Roadmap

- build a RAG tool that uses provided text file inputs and answers questions on that document
- build gradio app for simple prototype that interacts with user who provides document (file or directory) and query
- include semantic search to output the text locations for the search query, possibly with agent deciding between QA and Search
- write comprehensive tests
- adjust the tool such that it works for audio data, use model for transcription and a suitable embedding
- perhaps embed the audio itself and not only the speech transcription, e.g. to search for specific sounds in audio
- same steps for video inputs, extract audio and frames, each with its own embedding

## Features

Llama3, LlamaIndex, Gradio