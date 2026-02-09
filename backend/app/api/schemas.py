from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentCreateResponse(BaseModel):
    doc_id: str
    slug: str
    filename: str
    page_count: int
    token_count: int
    route: str
    index_version: str
    file_url: str


class DocumentInfo(BaseModel):
    doc_id: str
    slug: str
    filename: str
    page_count: int
    token_count: int
    route: str
    index_version: str
    file_url: str


class DocumentListResponse(BaseModel):
    docs: List[DocumentInfo]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    debug: bool = False


class Citation(BaseModel):
    doc_id: str
    source_id: str
    page_start: int
    page_end: int
    section_title: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    summary: Optional[str] = None
    citations: List[Citation]
    selected_doc_ids: List[str]
    route: str
    used_context_count: int
    debug: Optional[Dict[str, Any]] = None

