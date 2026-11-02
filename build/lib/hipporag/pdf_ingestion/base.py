from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChunkWithCitation:
    text: str
    source_file: str
    doc_title: str
    page_num: int
    section: Optional[str] = None

    @property
    def hash_id(self) -> str:
        from ..utils.misc_utils import compute_mdhash_id

        return compute_mdhash_id(self.text, prefix="chunk-")


class BasePDFParser(ABC):

    @abstractmethod
    def parse(self, pdf_path: str) -> List[ChunkWithCitation]:
        pass
