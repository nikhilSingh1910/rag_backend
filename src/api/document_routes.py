from flask import Blueprint, request, jsonify
from models.database import Document, DocumentEmbedding, engine
from sqlalchemy.orm import sessionmaker
from services.rag_service import RAGService
import json

document_bp = Blueprint('document', __name__)
Session = sessionmaker(bind=engine)
rag_service = RAGService()

@document_bp.route('/upload', methods=['POST'])
async def upload_document():
    try:
        data = request.get_json()
        title = data.get('title')
        content = data.get('content')
        
        if not title or not content:
            return jsonify({'error': 'Title and content are required'}), 400
        
        # Create document
        session = Session()
        document = Document(title=title, content=content)
        session.add(document)
        session.commit()
        
        # Process document with RAG service
        chunk_embeddings = await rag_service.process_document(content)
        
        # Save embeddings
        for chunk in chunk_embeddings:
            embedding = DocumentEmbedding(
                document_id=document.id,
                embedding=chunk['embedding'],
                chunk_text=chunk['chunk_text'],
                chunk_index=chunk['chunk_index']
            )
            session.add(embedding)
        
        session.commit()
        session.close()
        
        return jsonify({
            'message': 'Document uploaded and processed successfully',
            'document_id': document.id
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@document_bp.route('/list', methods=['GET'])
async def list_documents():
    try:
        session = Session()
        documents = session.query(Document).all()
        result = [{
            'id': doc.id,
            'title': doc.title,
            'created_at': doc.created_at.isoformat(),
            'updated_at': doc.updated_at.isoformat()
        } for doc in documents]
        session.close()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@document_bp.route('/<int:doc_id>', methods=['GET'])
async def get_document(doc_id):
    try:
        session = Session()
        document = session.query(Document).get(doc_id)
        
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        
        result = {
            'id': document.id,
            'title': document.title,
            'content': document.content,
            'created_at': document.created_at.isoformat(),
            'updated_at': document.updated_at.isoformat()
        }
        session.close()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 