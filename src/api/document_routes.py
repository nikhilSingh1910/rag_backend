from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.database import Document, DocumentEmbedding, engine
from sqlalchemy.orm import sessionmaker
from services.rag_service import RAGService
from utils.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    ValidationError,
    DatabaseError
)
from utils.schemas import DocumentCreate, DocumentResponse, ErrorResponse
from config.settings import settings
import logging
from typing import List, Dict
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

document_bp = Blueprint('document', __name__)
Session = sessionmaker(bind=engine)
rag_service = RAGService()
executor = ThreadPoolExecutor(max_workers=settings.rag_batch_size)

@document_bp.route('/upload', methods=['POST'])
@jwt_required()
async def upload_document():
    try:
        start_time = time.time()
        # Validate request data
        data = request.get_json()
        document_data = DocumentCreate(**data)
        
        # Get current user ID from JWT
        user_id = get_jwt_identity()
        
        session = Session()
        try:
            # Create document
            document = Document(
                title=document_data.title,
                content=document_data.content,
                user_id=user_id
            )
            session.add(document)
            session.commit()
            
            # Process document with RAG service
            chunk_embeddings = await rag_service.process_document(document_data.content)
            
            # Save embeddings in batches
            batch_size = 100
            for i in range(0, len(chunk_embeddings), batch_size):
                batch = chunk_embeddings[i:i + batch_size]
                embeddings = [
                    DocumentEmbedding(
                        document_id=document.id,
                        embedding=chunk['embedding'],
                        chunk_text=chunk['chunk_text'],
                        chunk_index=chunk['chunk_index']
                    ) for chunk in batch
                ]
                session.bulk_save_objects(embeddings)
                session.commit()
            
            processing_time = time.time() - start_time
            logger.info(f"Document uploaded and processed in {processing_time:.2f} seconds")
            
            response = DocumentResponse.from_orm(document)
            return jsonify(response.dict()), 201
            
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Database error: {str(e)}")
        finally:
            session.close()
            
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 400
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 500
    except DatabaseError as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify(ErrorResponse(error="Internal server error").dict()), 500

@document_bp.route('/batch-upload', methods=['POST'])
@jwt_required()
async def batch_upload_documents():
    try:
        start_time = time.time()
        # Validate request data
        data = request.get_json()
        if not isinstance(data, list):
            raise ValidationError("Request body must be a list of documents")
        
        # Get current user ID from JWT
        user_id = get_jwt_identity()
        
        session = Session()
        try:
            documents = []
            for doc_data in data:
                document_data = DocumentCreate(**doc_data)
                
                # Create document
                document = Document(
                    title=document_data.title,
                    content=document_data.content,
                    user_id=user_id
                )
                session.add(document)
                documents.append(document)
            
            session.commit()
            
            # Process documents in parallel
            async def process_document(doc):
                try:
                    chunk_embeddings = await rag_service.process_document(doc.content)
                    return doc.id, chunk_embeddings
                except Exception as e:
                    logger.error(f"Error processing document {doc.id}: {str(e)}")
                    return doc.id, None
            
            # Process documents concurrently
            tasks = [process_document(doc) for doc in documents]
            results = await asyncio.gather(*tasks)
            
            # Save embeddings in batches
            for doc_id, chunk_embeddings in results:
                if chunk_embeddings:
                    batch_size = 100
                    for i in range(0, len(chunk_embeddings), batch_size):
                        batch = chunk_embeddings[i:i + batch_size]
                        embeddings = [
                            DocumentEmbedding(
                                document_id=doc_id,
                                embedding=chunk['embedding'],
                                chunk_text=chunk['chunk_text'],
                                chunk_index=chunk['chunk_index']
                            ) for chunk in batch
                        ]
                        session.bulk_save_objects(embeddings)
                        session.commit()
            
            processing_time = time.time() - start_time
            logger.info(f"Batch upload completed in {processing_time:.2f} seconds")
            
            response = [DocumentResponse.from_orm(doc).dict() for doc in documents]
            return jsonify(response), 201
            
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Database error: {str(e)}")
        finally:
            session.close()
            
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 400
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 500
    except DatabaseError as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify(ErrorResponse(error="Internal server error").dict()), 500

@document_bp.route('/list', methods=['GET'])
@jwt_required()
async def list_documents():
    try:
        # Get current user ID from JWT
        user_id = get_jwt_identity()
        
        session = Session()
        try:
            documents = session.query(Document).filter_by(user_id=user_id).all()
            result = [DocumentResponse.from_orm(doc).dict() for doc in documents]
            return jsonify(result), 200
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return jsonify(ErrorResponse(error="Failed to list documents").dict()), 500

@document_bp.route('/<int:doc_id>', methods=['GET'])
@jwt_required()
async def get_document(doc_id):
    try:
        # Get current user ID from JWT
        user_id = get_jwt_identity()
        
        session = Session()
        try:
            document = session.query(Document).filter_by(id=doc_id, user_id=user_id).first()
            
            if not document:
                raise DocumentNotFoundError(f"Document with ID {doc_id} not found")
            
            response = DocumentResponse.from_orm(document)
            return jsonify(response.dict()), 200
            
        finally:
            session.close()
            
    except DocumentNotFoundError as e:
        logger.warning(f"Document not found: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 404
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        return jsonify(ErrorResponse(error="Failed to get document").dict()), 500

@document_bp.route('/<int:doc_id>', methods=['DELETE'])
@jwt_required()
async def delete_document(doc_id):
    try:
        # Get current user ID from JWT
        user_id = get_jwt_identity()
        
        session = Session()
        try:
            document = session.query(Document).filter_by(id=doc_id, user_id=user_id).first()
            
            if not document:
                raise DocumentNotFoundError(f"Document with ID {doc_id} not found")
            
            # Delete document and its embeddings
            session.delete(document)
            session.commit()
            
            return jsonify({"message": f"Document {doc_id} deleted successfully"}), 200
            
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Database error: {str(e)}")
        finally:
            session.close()
            
    except DocumentNotFoundError as e:
        logger.warning(f"Document not found: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 404
    except DatabaseError as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify(ErrorResponse(error=str(e)).dict()), 500
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return jsonify(ErrorResponse(error="Failed to delete document").dict()), 500 