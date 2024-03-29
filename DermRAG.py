from PyPDF2 import PdfReader
import pandas as pd 
import matplotlib.pyplot as plt
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from ragatouille import RAGPretrainedModel

import argparse
import warnings 
warnings.filterwarnings("ignore")



def split_documents(chunk_size: int, kb: list[LangchainDocument],DefSeparators: list,tokenizerName: str) -> list[LangchainDocument]:
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizerName),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=DefSeparators,
        device_map = "auto"
    )

    docs_processed = []
    for doc in kb:
        docs_processed += text_splitter.split_documents([doc])
    
    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def answer_with_rag(
    question: str,
    llm: pipeline,
    knowledge_index: FAISS,
    reranker,
    num_retrieved_docs: int = 3,
    num_docs_final: int = 2) -> tuple[str, list[LangchainDocument]]:


    # Gather documents with retriever
    print("===================> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    # Optionally rerank results
    if reranker:
        print("==================> Reranking documents...")
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    print("====================> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DermRAG')
    parser.add_argument("--query", type=str, default="How can I identify hypertrophic lupus erythematosus?")
    parser.add_argument("--em", type=str, default="thenlper/gte-small", help = "Embedding Model")
    parser.add_argument("--rm", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Reader model")
    parser.add_argument("--raw_database", type=str, default="project.pdf", help="path/name of the file")
    parser.add_argument("--reranker", type=str, default="colbert-ir/colbertv2.0", help="path/name of the file")

    args = parser.parse_args()   

    

    print(f'===================Loading Raw Data===================')
    pdf_path = args.raw_database
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PdfReader(pdf_file)

    RAW_KNOWLEDGE_BASE = [LangchainDocument(page_content=pdf_reader.pages[page].extract_text(), metadata={"source": page}) for page in range(11, len(pdf_reader.pages))]
    
    EMBEDDING_MODEL_NAME = args.em
    user_query = args.query
    sep = [".", "\n"]
    
    print(f'===================Splitting Documents===================')
    docs_processed = split_documents(512, RAW_KNOWLEDGE_BASE,DefSeparators = sep ,tokenizerName=EMBEDDING_MODEL_NAME)


    print(f'===================Initializing embedding model tokenizer===================')
    tokenizer_embed = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}  # set True for cosine similarity
    )

    print(f'===================Vectorizing Database===================')
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    print(f'===================Loading Reader Model==================')
    model_name = args.rm
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map = "auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = "auto")
    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        top_k = 4,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=150,
    )


    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context, give a comprehensive answer to the question. Respond only to the question asked, response should be concise and relevant to the question. Provide the number of the source document when relevant. If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]


    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    RERANKER = RAGPretrainedModel.from_pretrained(args.reranker)


    answer, relevant_docs = answer_with_rag(user_query, READER_LLM, KNOWLEDGE_VECTOR_DATABASE, reranker=RERANKER)

    print("==================================Answer==================================")
    print(f"{answer}")
