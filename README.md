# DermRAG

This repository contains code for DermRAG, a tool for generating answers to dermatology-related questions using a combination of language models and document retrieval techniques. DermRAG leverages advanced natural language processing (NLP) and machine learning (ML) models to provide comprehensive answers to user queries based on a given knowledge base of dermatological documents.

## Features
- **Document Splitting**: Utilizes the Langchain library to split documents into smaller chunks for efficient processing.
- **Document Vectorization**: Employs FAISS and embedding models from Huggingface for vectorizing documents in the knowledge base.
- **Embedding and Reader Models**: Allows users to specify custom embedding and reader models for fine-tuned performance.
- **Reranking**: Optionally reranks retrieved documents using pre-trained reranker models to improve answer relevance.
- **Question Answering**: Generates answers to user queries using a combination of transformer-based language models and document retrieval techniques.

## Requirements
- PyPDF2
- pandas
- matplotlib
- transformers
- torch
- langchain
- langchain_community
- ragatouille

## Usage
- Install the required dependencies.
- Clone the repository.
- Run the DermRAG script, specifying the necessary arguments such as  
  - Query: Question that you want to ask
  - Embedding model: The embedding model used to get the embeddings of chunks and questions
  - Reader model: The reader LLM which will generate the final answer
  - Raw database: The pdf file for the book you want your answers to be based on.
  - Reranker: Optional Reranker model to find the relevant chunks in the retrieval process

```bash
python DermRAG.py --query "How can I identify hypertrophic lupus erythematosus?" --em "thenlper/gte-small" --rm "meta-llama/Llama-2-7b-chat-hf" --raw_database "project.pdf" --reranker "colbert-ir/colbertv2.0"
