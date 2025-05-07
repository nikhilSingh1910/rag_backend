from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.database import Document, DocumentEmbedding, engine
from sqlalchemy.orm import sessionmaker
from services.rag_service import RAGService
from utils.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    ValidationError
)
from utils.schemas import QuestionRequest, QuestionResponse, ErrorResponse
import logging
from typing import List, Dict
import json
import time
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qa_bp = Blueprint('qa', __name__)
Session = sessionmaker(bind=engine)
rag_service = RAGService()

def log_performance(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            response = await f(*args, **kwargs)
            processing_time = time.time() - start_time
            logger.info(f"Request processed in {processing_time:.2f} seconds")
            return response
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Request failed after {processing_time:.2f} seconds: {str(e)}")
            raise
    return decorated_function

@qa_bp.route('/ask', methods=['POST'])
@jwt_required()
@log_performance
async def ask_question():
    try:
        # Validate request data
        data = request.get_json()
        question_request = QuestionRequest(**data)
        
        # Get current user ID from JWT
        user_id = get_jwt_identity()
        
        session = Session()
        try:
            # Get document embeddings with optimized query
            query = session.query(DocumentEmbedding).join(Document).filter(Document.user_id == user_id)
            if question_request.document_ids:
                query = query.filter(DocumentEmbedding.document_id.in_(question_request.document_ids))
            
            # Use limit to prevent memory issues with large datasets
            embeddings = query.limit(1000).all()
            
            if not embeddings:
                raise DocumentNotFoundError("No documents available for answering")
            
            # Prepare documents for RAG
            documents = [{
                'chunk_text': emb.chunk_text,
                'embedding': emb.embedding,
                'chunk_index': emb.chunk_index
            } for emb in embeddings]
            
            # Set up QA chain with selected documents
            await rag_service.setup_qa_chain(documents)
            
            # Get answer with performance monitoring
            result = await rag_service.get_answer(question_request.question)
            
            # Create response with performance metrics
            response = QuestionResponse(
                question=question_request.question,
                answer=result["answer"],
                relevant_documents=result["relevant_documents"],
                processing_time=result.get("processing_time", 0)
            )
            
            return jsonify(response.dict()), 200
            
        finally:
            session.close()
            
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 400
    except DocumentNotFoundError as e:
        logger.warning(f"Document not found: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 404
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify(ErrorResponse(error="Internal server error").dict()), 500

@qa_bp.route('/documents', methods=['GET'])
@jwt_required()
async def get_available_documents():
    try:
        # Get current user ID from JWT
        user_id = get_jwt_identity()
        
        session = Session()
        try:
            documents = session.query(Document).filter_by(user_id=user_id).all()
            result = [{
                'id': doc.id,
                'title': doc.title,
                'created_at': doc.created_at.isoformat()
            } for doc in documents]
            
            return jsonify(result), 200
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error getting available documents: {str(e)}")
        return jsonify(ErrorResponse(error="Failed to get available documents").dict()), 500

@qa_bp.route('/similar', methods=['POST'])
@jwt_required()
@log_performance
async def get_similar_chunks():
    try:
        # Validate request data
        data = request.get_json()
        if not data or 'query' not in data:
            raise ValidationError("Query is required")
        
        query = data['query']
        k = min(data.get('k', 3), 10)  # Limit k to prevent performance issues
        
        # Get current user ID from JWT
        user_id = get_jwt_identity()
        
        session = Session()
        try:
            # Get document embeddings with optimized query
            embeddings = session.query(DocumentEmbedding).join(Document).filter(
                Document.user_id == user_id
            ).limit(1000).all()  # Limit to prevent memory issues
            
            if not embeddings:
                raise DocumentNotFoundError("No documents available")
            
            # Prepare documents for RAG
            documents = [{
                'chunk_text': emb.chunk_text,
                'embedding': emb.embedding,
                'chunk_index': emb.chunk_index
            } for emb in embeddings]
            
            # Set up QA chain
            await rag_service.setup_qa_chain(documents)
            
            # Get similar chunks with performance monitoring
            similar_chunks = await rag_service.get_similar_chunks(query, k=k)
            
            return jsonify(similar_chunks), 200
            
        finally:
            session.close()
            
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 400
    except DocumentNotFoundError as e:
        logger.warning(f"Document not found: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 404
    except Exception as e:
        logger.error(f"Error getting similar chunks: {str(e)}")
        return jsonify(ErrorResponse(error="Failed to get similar chunks").dict()), 500

@qa_bp.route('/clear-cache', methods=['POST'])
@jwt_required()
async def clear_cache():
    """Clear the RAG service cache"""
    try:
        rag_service.clear_cache()
        return jsonify({"message": "Cache cleared successfully"}), 200
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify(ErrorResponse(error="Failed to clear cache").dict()), 500 