from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.database import get_db
from app.services.document_service import DocumentService
from app.db.models import Document
from pydantic import BaseModel

router = APIRouter()
document_service = DocumentService()

class DocumentCreate(BaseModel):
    title: str
    content: str
    file_type: Optional[str] = None

class DocumentResponse(BaseModel):
    id: int
    title: str
    file_type: Optional[str]
    created_at: str

    class Config:
        from_attributes = True

@router.post("/documents/", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreate,
    db: Session = Depends(get_db)
):
    try:
        doc = await document_service.process_document(
            db=db,
            title=document.title,
            content=document.content,
            file_type=document.file_type
        )
        return doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/upload/", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    content = await file.read()
    content_str = content.decode("utf-8")
    
    try:
        doc = await document_service.process_document(
            db=db,
            title=file.filename,
            content=content_str,
            file_path=file.filename,
            file_type=file.content_type
        )
        return doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    documents = await document_service.get_documents(db, skip=skip, limit=limit)
    return documents

@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    document = await document_service.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    success = await document_service.delete_document(db, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"} 