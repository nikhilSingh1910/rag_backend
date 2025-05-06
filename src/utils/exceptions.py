class RAGException(Exception):
    """Base exception for RAG-related errors"""
    pass

class DocumentProcessingError(RAGException):
    """Raised when there's an error processing a document"""
    pass

class EmbeddingGenerationError(RAGException):
    """Raised when there's an error generating embeddings"""
    pass

class DocumentNotFoundError(RAGException):
    """Raised when a document is not found"""
    pass

class InvalidDocumentError(RAGException):
    """Raised when document data is invalid"""
    pass

class AuthenticationError(RAGException):
    """Raised when there's an authentication error"""
    pass

class DatabaseError(RAGException):
    """Raised when there's a database error"""
    pass

class ValidationError(RAGException):
    """Raised when input validation fails"""
    pass 