from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import json
from typing import List, Dict
import numpy as np

class RAGService:
    def __init__(self):
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize LLM
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )
        
        self.vector_store = None

    async def process_document(self, content: str) -> List[Dict]:
        """Process a document and return chunks with embeddings"""
        # Split text into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Generate embeddings for each chunk
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.embed_query(chunk)
            chunk_embeddings.append({
                "chunk_text": chunk,
                "chunk_index": i,
                "embedding": json.dumps(embedding.tolist())
            })
        
        return chunk_embeddings

    async def setup_qa_chain(self, documents: List[Dict]):
        """Set up the QA chain with the provided documents"""
        texts = [doc["chunk_text"] for doc in documents]
        embeddings = [json.loads(doc["embedding"]) for doc in documents]
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_texts(
            texts,
            self.embeddings,
            metadatas=[{"source": i} for i in range(len(texts))]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
        )
        
        return qa_chain

    async def get_answer(self, question: str) -> str:
        """Get answer for a question using the QA chain"""
        if not self.vector_store:
            raise ValueError("QA chain not initialized. Please set up documents first.")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
        )
        
        response = qa_chain.run(question)
        return response 