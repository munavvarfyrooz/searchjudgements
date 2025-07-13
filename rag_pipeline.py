import os
import json
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import torch
from utils import get_pdf_paths, extract_text_from_pdf
import multiprocessing as mp

# Constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "microsoft/phi-2"
FAISS_INDEX_PATH = "./faiss_index/"
PDF_DIR = "./judgements_pdfs/"
BATCH_SIZE = 10

def build_index(pdf_dir: str = PDF_DIR, faiss_path: str = FAISS_INDEX_PATH) -> None:
    """
    Build the FAISS index from PDFs in the given directory.
    Ingests in batches with multiprocessing for efficiency.
    
    Args:
        pdf_dir (str): Directory containing PDFs.
        faiss_path (str): Path to save FAISS index.
    """
    try:
        pdf_paths = get_pdf_paths(pdf_dir)
        if not pdf_paths:
            raise ValueError(f"No PDFs found in {pdf_dir}")
            
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = None
        num_cores = mp.cpu_count()
        pool = mp.Pool(processes=num_cores)
        
        for batch_start in range(0, len(pdf_paths), BATCH_SIZE):
            batch = pdf_paths[batch_start:batch_start + BATCH_SIZE]
            with tqdm(total=len(batch), desc="Processing PDF batch") as pbar:
                results = []
                for pdf in batch:
                    results.append(pool.apply_async(_process_pdf, (pdf,), callback=lambda x: pbar.update(1)))
                documents = []
                for res in results:
                    documents.extend(res.get())
            if vectorstore is None:
                vectorstore = FAISS.from_documents(documents, embeddings)
            else:
                vectorstore.add_documents(documents)
        pool.close()
        pool.join()
        vectorstore.save_local(faiss_path)
        print(f"FAISS index saved to {faiss_path}")
    except Exception as e:
        print(f"Error building index: {e}")
def _process_pdf(pdf_path: str) -> List[Document]:
    """Helper function for multiprocessing PDF processing."""
    text = extract_text_from_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks]

def load_vectorstore(faiss_path: str = FAISS_INDEX_PATH) -> FAISS:
    """Load existing FAISS vectorstore."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise ValueError(f"Error loading FAISS index from {faiss_path}: {e}")

def hybrid_retrieve(query: str, vectorstore: FAISS, top_k: int = 10) -> List[Document]:
    """
    Hybrid retrieval: semantic + BM25, rerank top results.
    
    Args:
        query (str): User query.
        vectorstore (FAISS): Loaded vectorstore.
        top_k (int): Number of top results to return.
    
    Returns:
        List[Document]: Reranked relevant documents.
    """
    try:
        # Semantic retriever
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k * 2})
        
        # BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(vectorstore.docstore.values())
        bm25_retriever.k = top_k * 2
        
        # Ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        # Retrieve and rerank (simple score-based rerank)
        results = ensemble_retriever.get_relevant_documents(query)
        # Rerank by ensemble score (assuming scores are available; fallback to sorting by metadata if needed)
        results.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"Error in hybrid retrieval: {e}")
        return []

def query_rag(query: str, vectorstore: FAISS) -> Tuple[str, List[str]]:
    """
    Query the RAG system and generate response using local LLM.
    
    Args:
        query (str): User query.
        vectorstore (FAISS): Loaded vectorstore.
    
    Returns:
        Tuple[str, List[str]]: Generated response and list of sources.
    """
    try:
        contexts = hybrid_retrieve(query, vectorstore)
        context_text = "\n\n".join([doc.page_content for doc in contexts])
        sources = [doc.metadata["source"] for doc in contexts]
        
        # Prompt template
        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""You are a legal expert summarizing judgements. Based on the following context, answer the query. 
            Summarize key points, cite sources accurately, and avoid hallucinations. If information is insufficient, say so.
            
            Query: {query}
            
            Context: {context}
            
            Response:"""
        )
        
        # Load local LLM
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, trust_remote_code=True)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, device=0 if torch.cuda.is_available() else -1)
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Generate response
        prompt = prompt_template.format(query=query, context=context_text)
        response = llm(prompt)
        
        return response, sources
    except Exception as e:
        print(f"Error querying RAG: {e}")
        return "An error occurred.", []
