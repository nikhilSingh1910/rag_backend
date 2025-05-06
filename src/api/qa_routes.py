from flask import Blueprint, request, jsonify
from models.database import Document, DocumentEmbedding, engine
from sqlalchemy.orm import sessionmaker
from services.rag_service import RAGService
import json

qa_bp = Blueprint('qa', __name__)
Session = sessionmaker(bind=engine)
rag_service = RAGService()

@qa_bp.route('/ask', methods=['POST'])
async def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')
        document_ids = data.get('document_ids', [])  # Optional list of document IDs to consider
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        session = Session()
        
        # Get document embeddings
        query = session.query(DocumentEmbedding)
        if document_ids:
            query = query.filter(DocumentEmbedding.document_id.in_(document_ids))
        
        embeddings = query.all()
        
        if not embeddings:
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
async def get_available_documents():
    try:
        session = Session()
        documents = session.query(Document).all()
        result = [{
            'id': doc.id,
            'title': doc.title,
            'created_at': doc.created_at.isoformat()
        } for doc in documents]
        session.close()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 