from dataclasses import dataclass
from typing import List, Optional, Tuple

Triple = Tuple[str, str, str]


@dataclass
class Citation:
    source_file: str
    doc_title: str
    page_num: int
    section: Optional[str] = None
    text_snippet: str = ""
    # Все документы, содержащие этот фрагмент (S-12/D-1): текст чанка может
    # встречаться в нескольких PDF, и цитата обязана показывать каждый из них
    sources: Optional[List[str]] = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = [self.source_file] if self.source_file else []


@dataclass
class QuerySolution:
    question: str = ""
    docs: List[str] = None
    doc_scores: List[float] = None
    citations: List[Citation] = None
    answer: str = ""
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None

    def __post_init__(self):
        if self.docs is None:
            self.docs = []
        if self.doc_scores is None:
            self.doc_scores = []
        if self.citations is None:
            self.citations = []

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": list(self.docs)[:5],
            "doc_scores": (
                [round(float(v), 4) for v in list(self.doc_scores)[:5]]
                if self.doc_scores is not None
                else None
            ),
            "gold_docs": self.gold_docs,
        }
