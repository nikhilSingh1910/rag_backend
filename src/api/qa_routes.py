from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.database import Document, DocumentEmbedding, engine
from sqlalchemy.orm import sessionmaker
from services.rag_service import RAGService
import json

qa_bp = Blueprint('qa', __name__)
Session = sessionmaker(bind=engine)
rag_service = RAGService()

@qa_bp.route('/ask', methods=['POST'])
@jwt_required()
async def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')
        document_ids = data.get('document_ids', [])  # Optional list of document IDs to consider
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Get current user ID from JWT
        user_id = get_jwt_identity()
        
        session = Session()
        
        # Get document embeddings
        query = session.query(DocumentEmbedding).join(Document).filter(Document.user_id == user_id)
        if document_ids:
            query = query.filter(DocumentEmbedding.document_id.in_(document_ids))
        
        embeddings = query.all()
        
        if not embeddings:
            session.close()
            return jsonify({'error': 'No documents available for answering'}), 400
        
        # Prepare documents for RAG
        documents = [{
            'chunk_text': emb.chunk_text,
            'embedding': emb.embedding,
            'chunk_index': emb.chunk_index
        } for emb in embeddings]
        
        # Set up QA chain with selected documents
        await rag_service.setup_qa_chain(documents)
        
        # Get answer
        answer = await rag_service.get_answer(question)
        
        session.close()
        
        return jsonify({
            'question': question,
            'answer': answer
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@qa_bp.route('/documents', methods=['GET'])
@jwt_required()
async def get_available_documents():
    try:
        # Get current user ID from JWT
        user_id = get_jwt_identity()
        
        session = Session()
        documents = session.query(Document).filter_by(user_id=user_id).all()
        result = [{
            'id': doc.id,
            'title': doc.title,
            'created_at': doc.created_at.isoformat()
        } for doc in documents]
        session.close()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 