from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.database import get_db
from app.services.qa_service import QAService
from pydantic import BaseModel

router = APIRouter()
qa_service = QAService()

class QuestionRequest(BaseModel):
    question: str
    document_ids: Optional[List[int]] = None

class SourceResponse(BaseModel):
    document_id: int
    content: str
    chunk_index: int

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]

@router.post("/qa/", response_model=str)
async def get_answer(
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    try:
        answer = await qa_service.get_answer(
            db=db,
            question=request.question,
            document_ids=request.document_ids
        )
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/qa/with-sources", response_model=AnswerResponse)
async def get_answer_with_sources(
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    try:
        response = await qa_service.get_answer_with_sources(
            db=db,
            question=request.question,
            document_ids=request.document_ids
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 