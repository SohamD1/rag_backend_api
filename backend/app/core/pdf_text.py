from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass(frozen=True)
class PageText:
    page_num: int
    text: str


def extract_pages(pdf_path: str) -> List[PageText]:
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    pages: List[PageText] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(PageText(page_num=i + 1, text=text))
    return pages

