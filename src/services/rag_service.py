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
            
            # Initialize text splitter with optimized parameters
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
            logger.info("RAGService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAGService: {str(e)}")
            raise DocumentProcessingError(f"Failed to initialize RAG service: {str(e)}")

    async def process_document(self, content: str) -> List[Dict]:
        """Process a document and return chunks with embeddings"""
        try:
            logger.info("Starting document processing")
            
            # Validate input
            if not content or not isinstance(content, str):
                raise ValidationError("Invalid document content")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            logger.info(f"Document split into {len(chunks)} chunks")
            
            # Generate embeddings for each chunk
            chunk_embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.embeddings.embed_query(chunk)
                    chunk_embeddings.append({
                        "chunk_text": chunk,
                        "chunk_index": i,
                        "embedding": json.dumps(embedding.tolist())
                    })
                except Exception as e:
                    logger.error(f"Error generating embedding for chunk {i}: {str(e)}")
                    raise EmbeddingGenerationError(f"Failed to generate embedding: {str(e)}")
            
            logger.info("Document processing completed successfully")
            return chunk_embeddings
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise DocumentProcessingError(f"Failed to process document: {str(e)}")

    async def setup_qa_chain(self, documents: List[Dict], k: int = 3) -> None:
        """Set up the QA chain with the provided documents"""
        try:
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
            
            logger.info("QA chain setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            raise DocumentProcessingError(f"Failed to set up QA chain: {str(e)}")

    async def get_answer(self, question: str, k: int = 3) -> Dict:
        """Get answer for a question using the QA chain"""
        try:
            if not self.vector_store:
                raise ValidationError("QA chain not initialized. Please set up documents first.")
            
            # Create QA chain with custom prompt
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
            
            return {
                "answer": result["result"],
                "relevant_documents": relevant_docs
            }
            
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            raise DocumentProcessingError(f"Failed to get answer: {str(e)}")

    async def get_similar_chunks(self, query: str, k: int = 3) -> List[Dict]:
        """Get similar chunks for a query"""
        try:
            if not self.vector_store:
                raise ValidationError("Vector store not initialized")
            
            # Get similar documents
            docs = self.vector_store.similarity_search_with_score(query, k=k)
            
            return [{
                "text": doc[0].page_content,
                "score": doc[1],
                "metadata": doc[0].metadata
            } for doc in docs]
            
        except Exception as e:
            logger.error(f"Error getting similar chunks: {str(e)}")
            raise DocumentProcessingError(f"Failed to get similar chunks: {str(e)}") 