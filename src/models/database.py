from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/rag_db')

engine = create_engine(DATABASE_URL)
Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with embeddings
    embeddings = relationship("DocumentEmbedding", back_populates="document")

class DocumentEmbedding(Base):
    __tablename__ = 'document_embeddings'

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    embedding = Column(Text, nullable=False)  # Store as JSON string
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # Relationship with document
    document = relationship("Document", back_populates="embeddings")

# Create tables
def init_db():
    Base.metadata.create_all(engine) 