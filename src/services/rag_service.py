from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import json
from typing import List, Dict, Optional
import numpy as np
import logging
from functools import lru_cache
import time
from utils.exceptions import (
    DocumentProcessingError,
    EmbeddingGenerationError,
    ValidationError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        try:
            # Initialize the embedding model with caching
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize text splitter with optimized parameters for better chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize LLM with optimized parameters
            self.llm = HuggingFaceHub(
                repo_id="google/flan-t5-base",
                model_kwargs={
                    "temperature": 0.5,
                    "max_length": 512,
                    "top_p": 0.95,
                    "do_sample": True
                }
            )
            
            # Custom prompt template for better answers
            self.qa_prompt = PromptTemplate(
                template="""You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                
                Context: {context}
                
                Question: {question}
                
                Answer:""",
                input_variables=["context", "question"]
            )
            
            self.vector_store = None
            self._cache = {}
            logger.info("RAGService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAGService: {str(e)}")
            raise DocumentProcessingError(f"Failed to initialize RAG service: {str(e)}")

    @lru_cache(maxsize=1000)
    async def process_document(self, content: str) -> List[Dict]:
        """Process a document and return chunks with embeddings"""
        try:
            start_time = time.time()
            logger.info("Starting document processing")
            
            # Validate input
            if not content or not isinstance(content, str):
                raise ValidationError("Invalid document content")
            
            # Split text into chunks with optimized parameters
            chunks = self.text_splitter.split_text(content)
            logger.info(f"Document split into {len(chunks)} chunks")
            
            # Generate embeddings for each chunk with batch processing
            chunk_embeddings = []
            batch_size = 32  # Process in batches for better performance
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                try:
                    # Generate embeddings for the batch
                    batch_embeddings = self.embeddings.embed_documents(batch)
                    
                    for j, embedding in enumerate(batch_embeddings):
                        chunk_embeddings.append({
                            "chunk_text": batch[j],
                            "chunk_index": i + j,
                            "embedding": json.dumps(embedding.tolist())
                        })
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch {i//batch_size}: {str(e)}")
                    raise EmbeddingGenerationError(f"Failed to generate embeddings: {str(e)}")
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            return chunk_embeddings
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise DocumentProcessingError(f"Failed to process document: {str(e)}")

    async def setup_qa_chain(self, documents: List[Dict], k: int = 3) -> None:
        """Set up the QA chain with the provided documents"""
        try:
            start_time = time.time()
            logger.info("Setting up QA chain")
            
            texts = [doc["chunk_text"] for doc in documents]
            embeddings = [json.loads(doc["embedding"]) for doc in documents]
            
            # Create FAISS vector store with optimized parameters
            self.vector_store = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=[{"source": i} for i in range(len(texts))],
                normalize_L2=True
            )
            
            setup_time = time.time() - start_time
            logger.info(f"QA chain setup completed in {setup_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            raise DocumentProcessingError(f"Failed to set up QA chain: {str(e)}")

    async def get_answer(self, question: str, k: int = 3) -> Dict:
        """Get answer for a question using the QA chain"""
        try:
            start_time = time.time()
            
            if not self.vector_store:
                raise ValidationError("QA chain not initialized. Please set up documents first.")
            
            # Check cache first
            cache_key = f"{question}_{k}"
            if cache_key in self._cache:
                logger.info("Retrieved answer from cache")
                return self._cache[cache_key]
            
            # Create QA chain with custom prompt and optimized parameters
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={
                        "k": k,
                        "score_threshold": 0.5
                    }
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.qa_prompt}
            )
            
            # Get answer and relevant documents
            result = qa_chain({"query": question})
            
            # Extract relevant document IDs
            relevant_docs = [doc.metadata["source"] for doc in result["source_documents"]]
            
            response = {
                "answer": result["result"],
                "relevant_documents": relevant_docs,
                "processing_time": time.time() - start_time
            }
            
            # Cache the result
            self._cache[cache_key] = response
            logger.info(f"Generated answer in {response['processing_time']:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            raise DocumentProcessingError(f"Failed to get answer: {str(e)}")

    async def get_similar_chunks(self, query: str, k: int = 3) -> List[Dict]:
        """Get similar chunks for a query"""
        try:
            start_time = time.time()
            
            if not self.vector_store:
                raise ValidationError("Vector store not initialized")
            
            # Check cache first
            cache_key = f"similar_{query}_{k}"
            if cache_key in self._cache:
                logger.info("Retrieved similar chunks from cache")
                return self._cache[cache_key]
            
            # Get similar documents with optimized search
            docs = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                score_threshold=0.5
            )
            
            results = [{
                "text": doc[0].page_content,
                "score": doc[1],
                "metadata": doc[0].metadata
            } for doc in docs]
            
            # Cache the results
            self._cache[cache_key] = results
            logger.info(f"Retrieved similar chunks in {time.time() - start_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting similar chunks: {str(e)}")
            raise DocumentProcessingError(f"Failed to get similar chunks: {str(e)}")

    def clear_cache(self):
        """Clear the service cache"""
        self._cache.clear()
        logger.info("Cache cleared") 