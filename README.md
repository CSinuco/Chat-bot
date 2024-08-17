# Conversational Assistant Math PDF - PDF Based QA System

This project creates a conversational assistant that can answer questions based on the content of PDF documents. The assistant uses an OpenSource Large Language Model (LLM) from the Mistral family, combined with document processing and retrieval capabilities provided by LangChain and HuggingFace.

## Project Overview

The goal of this project is to develop a personal virtual assistant that leverages a knowledge base created from PDF documents. The assistant can handle questions by retrieving relevant information from the PDFs and generating responses using an LLM. Key concepts include system analysis, knowledge management, and the integration of various open-source tools to achieve a coherent, interactive assistant.

## Poster
<p align="center">
<img src="https://github.com/Andrew552004/project-docker/blob/main/DmKknh-4T3Wo3WDpfARaOQ.webp" alt="FastAPI logo" width="600">
</p>
<p align="center">
  <a href="https://ideogram.ai/t/explore">Created with Ideogram.ai</a>
</p>  
  
## System Description

1. **PDF Data Loading**:
   - Loads all PDF files from a specified directory into a single text string.

2. **Text Splitting**:
   - Splits the loaded text into manageable chunks of 1000 characters with a 50-character overlap to maintain context coherence.

3. **Embedding Generation**:
   - Uses HuggingFace's sentence-transformers to generate semantic embeddings for the text chunks.

4. **Vector Store Creation**:
   - Creates a FAISS vector store from the embeddings to enable efficient similarity search.

5. **LLM Integration**:
   - Integrates a Mistral-family LLM using LlamaCpp for generating responses based on the retrieved text chunks.

6. **Question Answering**:
   - Implements a retrieval-based QA system using LangChain's RetrievalQA to answer questions by searching the vector store and generating responses with the LLM.

## How to Use

Follow these steps to set up and use the project:
1. **Make sure you have poetry installed***

2. **Copy this for create the virtual enviroments**
   ```bash
   python -m venv chat_bot
   ```
3. **Activate the virtual environment**:
   ```bash
   chat_bot\Scripts\Activate.ps1
   ```
4. **Put this for generate the poetry.lock**
   ```bash
   poetry lock
   ```
5. **Put this for install the dependencies in the enviroment**
   ```bash
   poetry install
   ```
6. **To run the code in the terminal run the following script**
   ```bash
   python .\your_assistant\text_processing.py
   ```
7. **To use the data container documents, you can find them in the following Drive link** 
   ```bash
   https://drive.google.com/drive/folders/1cczSAUuN_z9bBLNMGH78lOH9XInLwVQE?usp=drive_link
   ```
