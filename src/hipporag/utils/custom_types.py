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
    # H2-6: только те цитаты, на которые модель реально сослалась, плюс
    # признаки отказа и ошибок заземления. Поля заполняет HippoRAG.qa().
    used_citations: List[Citation] = None
    abstained: bool = False
    grounding_errors: List[str] = None

    def __post_init__(self):
        if self.docs is None:
            self.docs = []
        if self.doc_scores is None:
            self.doc_scores = []
        if self.citations is None:
            self.citations = []
        if self.used_citations is None:
            self.used_citations = []
        if self.grounding_errors is None:
            self.grounding_errors = []

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "grounding_errors": self.grounding_errors,
            "gold_answers": self.gold_answers,
            "docs": list(self.docs)[:5],
            "doc_scores": (
                [round(float(v), 4) for v in list(self.doc_scores)[:5]]
                if self.doc_scores is not None
                else None
            ),
            "gold_docs": self.gold_docs,
        }
