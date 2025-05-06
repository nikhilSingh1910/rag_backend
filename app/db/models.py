from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    file_path = Column(String(512), nullable=True)
    file_type = Column(String(10), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"))
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(Text, nullable=False)  # Store embedding as JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# Association table for document selection
document_selection = Table(
    "document_selection",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE")),
    Column("document_id", Integer, ForeignKey("documents.id", ondelete="CASCADE")),
) 