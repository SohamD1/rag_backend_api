from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentCreateResponse(BaseModel):
    doc_id: str
    slug: str
    filename: str
    source_url: str
    page_count: int
    token_count: int
    route: str
    index_version: str
    file_url: Optional[str] = None


class DocumentInfo(BaseModel):
    doc_id: str
    slug: str
    filename: str
    source_url: str
    page_count: int
    token_count: int
    route: str
    index_version: str
    file_url: Optional[str] = None


class DocumentListResponse(BaseModel):
    docs: List[DocumentInfo]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    debug: bool = False


class Citation(BaseModel):
    doc_id: str
    filename: Optional[str] = None
    file_url: Optional[str] = None
    source_url: str
    source_id: str
    page_start: int
    page_end: int
    section_title: Optional[str] = None


class RetrievedChunk(BaseModel):
    source_id: str
    doc_id: str
    filename: Optional[str] = None
    file_url: Optional[str] = None
    source_url: Optional[str] = None
    text: str
    score: float
    page_start: int
    page_end: int
    section_title: Optional[str] = None
    route: str


class ChatResponse(BaseModel):
    answer: str
    summary: Optional[str] = None
    citations: List[Citation]
    chunks: Optional[List[RetrievedChunk]] = None
    selected_doc_ids: List[str]
    route: str
    used_context_count: int
    debug: Optional[Dict[str, Any]] = None
