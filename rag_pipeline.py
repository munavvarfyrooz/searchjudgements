import os
import json
import logging
from typing import List, Tuple, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_xai import ChatXAI
from tqdm import tqdm
from utils import get_pdf_paths, extract_text_from_pdf
import multiprocessing as mp
import concurrent.futures
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
XAI_MODEL = "grok-4"  # Valid xAI model name
BM25_INDEX_PATH = "bm25_index.pkl"
FAISS_INDEX_PATH = "./faiss_index/"
PDF_DIR = "./judgements_pdfs/"
BATCH_SIZE = 10
PROCESS_TIMEOUT = 300  # 5 minutes timeout for processing each PDF

def build_index(pdf_input: Union[str, List[str]] = PDF_DIR, faiss_path: str = FAISS_INDEX_PATH) -> None:
    """
    Build the FAISS index from PDFs in the given directory or list of paths.
    Ingests in batches with multiprocessing for efficiency.
    
    Args:
        pdf_input (Union[str, List[str]]): Directory containing PDFs or list of PDF paths.
        faiss_path (str): Path to save FAISS index.
    """
    try:
        if isinstance(pdf_input, str):
            pdf_paths = get_pdf_paths(pdf_input)
        elif isinstance(pdf_input, list):
            pdf_paths = [p for p in pdf_input if os.path.isfile(p) and p.lower().endswith('.pdf')]
        else:
            raise ValueError("pdf_input must be str (directory) or List[str] (paths)")
        
        if not pdf_paths:
            raise ValueError(f"No PDFs found in {pdf_input}")
        
        logger.info(f"Found {len(pdf_paths)} PDFs to process")
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = None
        num_cores = mp.cpu_count()
        with mp.Pool(processes=num_cores) as pool:
            for batch_start in range(0, len(pdf_paths), BATCH_SIZE):
                batch = pdf_paths[batch_start:batch_start + BATCH_SIZE]
                with tqdm(total=len(batch), desc="Processing PDF batch") as pbar:
                    results = []
                    for pdf in batch:
                        results.append(pool.apply_async(_process_pdf, (pdf,), callback=lambda x: pbar.update(1), error_callback=lambda e: logger.error(f"Error processing {pdf}: {e}")))
                    documents = []
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_to_res = {executor.submit(res.get, timeout=PROCESS_TIMEOUT): res for res in results}
                        for future in concurrent.futures.as_completed(future_to_res):
                            try:
                                documents.extend(future.result())
                            except concurrent.futures.TimeoutError:
                                logger.error(f"Timeout processing PDF")
                            except Exception as e:
                                logger.error(f"Error in batch processing: {e}")
                if not documents:
                    logger.warning("No documents processed in this batch")
                    continue
                
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(documents, embeddings)
                else:
                    vectorstore.add_documents(documents)
                logger.info(f"Added {len(documents)} chunks to vectorstore")
        
        if vectorstore is None:
            raise ValueError("No documents were processed - vectorstore is None")
        
        vectorstore.save_local(faiss_path)
        logger.info(f"FAISS index saved to {faiss_path}")
        
        # Create and save BM25 index for fast querying
        docs = list(vectorstore.docstore._dict.values())
        if docs:
            bm25_retriever = BM25Retriever.from_documents(docs)
            with open(BM25_INDEX_PATH, "wb") as f:
                pickle.dump(bm25_retriever, f)
            logger.info(f"BM25 index saved to {BM25_INDEX_PATH}")
        else:
            logger.warning("No documents to create BM25 index")
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise

def _process_pdf(pdf_path: str) -> List[Document]:
    """Helper function for multiprocessing PDF processing."""
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            logger.warning(f"Empty text extracted from {pdf_path}")
            return []
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(text)
        return [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks]
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        return []

def load_vectorstore(faiss_path: str = FAISS_INDEX_PATH) -> FAISS:
    """Load existing FAISS vectorstore."""
    try:
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}. Please build the index first.")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Error loading FAISS index from {faiss_path}: {e}")
        raise

def hybrid_retrieve(query: str, vectorstore: FAISS, top_k: int = 4) -> List[Document]:
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
        if vectorstore is None:
            logger.warning("Vectorstore is None - returning empty list")
            return []
        
        # Semantic retriever
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k * 2})
        
        # BM25 retriever - load precomputed index
        try:
            with open(BM25_INDEX_PATH, "rb") as f:
                bm25_retriever = pickle.load(f)
            bm25_retriever.k = top_k * 2
            logger.info("Loaded precomputed BM25 index")
        except FileNotFoundError:
            logger.warning("BM25 index not found, creating on the fly")
            docs = list(vectorstore.docstore._dict.values())
            if not docs:
                logger.warning("Docstore is empty - returning empty list")
                return []
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = top_k * 2
            with open(BM25_INDEX_PATH, "wb") as f:
                pickle.dump(bm25_retriever, f)
            logger.info(f"BM25 index created and saved to {BM25_INDEX_PATH}")
        
        # Ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        # Retrieve and rerank (simple score-based rerank)
        results = ensemble_retriever.invoke(query)
        logger.info(f"Retrieved {len(results)} documents for query")
        results.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        return results[:top_k]
    except Exception as e:
        logger.error(f"Error in hybrid retrieval: {e}")
        return []

def query_rag(query: str, vectorstore: FAISS) -> Tuple[str, List[str]]:
    """Query the RAG system."""
    try:
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable not set")
        
        llm = ChatXAI(
            api_key=api_key,
            model=XAI_MODEL,
            temperature=0.0,
            max_tokens=4096,
            top_p=1.0,
            stream=False
        )
        
        PROMPT = PromptTemplate(
            input_variables=["query", "context"],
            template="""You are a legal expert summarizing judgements. Based on the following context, answer the query.
Summarize key points, cite sources accurately, and avoid hallucinations. If information is insufficient, say so.
            
Query: {query}
            
Context: {context}
            
Response:"""
        )
        
        # Create hybrid retriever
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        
        # BM25 retriever - load or create
        try:
            with open(BM25_INDEX_PATH, "rb") as f:
                bm25_retriever = pickle.load(f)
            bm25_retriever.k = 20
            logger.info("Loaded precomputed BM25 index")
        except FileNotFoundError:
            logger.warning("BM25 index not found, creating on the fly")
            docs = list(vectorstore.docstore._dict.values())
            if not docs:
                logger.warning("Docstore is empty - returning empty list")
                return "", []
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 20
            with open(BM25_INDEX_PATH, "wb") as f:
                pickle.dump(bm25_retriever, f)
            logger.info(f"BM25 index created and saved to {BM25_INDEX_PATH}")
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        result = qa_chain.invoke({"query": query})
        logger.info(f"Retrieved {len(result['source_documents'])} documents for query")
        logger.info(f"Generated prompt (first 200 chars): {PROMPT.template[:200]}...")
        
        # Enhanced response handling for empty content
        response = result['result']
        if not response:
            logger.warning("LLM returned empty response")
            if hasattr(result, 'additional_kwargs') and 'reasoning_content' in result.additional_kwargs:
                response = result.additional_kwargs['reasoning_content']
                logger.info("Falling back to reasoning_content")
        
        return response, [doc.metadata["source"] for doc in result['source_documents']]
    except Exception as e:
        logger.error(f"Error in query_rag: {e}")
        return "", []
