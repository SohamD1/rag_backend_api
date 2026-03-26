from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    page_num: int
    text: str

