from __future__ import annotations

import logging
import os
from typing import List, Optional

from .base import ChunkWithCitation

logger = logging.getLogger(__name__)


class TextPDFParser:

    def __init__(self, chunk_config=None):
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF not installed. Install with: pip install PyMuPDF")

        self.fitz = fitz

        if chunk_config is None:
            self.ChunkConfig = None
            self.ChunkWithCitation = ChunkWithCitation
        else:
            self.ChunkConfig = chunk_config.get("ChunkConfig", None)
            self.ChunkWithCitation = chunk_config.get("ChunkWithCitation", None)

        if self.ChunkConfig is None or self.ChunkWithCitation is None:
            if not self.ChunkConfig:
                self.ChunkConfig = None
            if not self.ChunkWithCitation:
                self.ChunkWithCitation = ChunkWithCitation

    def _extract_title_from_path(self, pdf_path: str) -> str:
        filename = os.path.basename(pdf_path)
        title = os.path.splitext(filename)[0]

        replacements = {
            "_": " ",
            "-": " ",
        }
        for old, new in replacements.items():
            title = title.replace(old, new)

        return title.title()

    def _detect_section(
        self, page_text: str, current_page_num: int, all_pages: list
    ) -> Optional[str]:
        lines = page_text.strip().split("\n")
        for line in lines[:5]:
            stripped = line.strip()
            if len(stripped) > 10 and len(stripped) < 100:
                if any(
                    keyword in stripped.lower()
                    for keyword in ["chapter", "глав", "раздел", "section", "part", "часть"]
                ):
                    return stripped

            if current_page_num == 1 and len(stripped) > 20 and len(stripped) < 200:
                if any(
                    word in stripped.lower()
                    for word in ["astrolog", "planet", "zodiac", "гороскоп", "планет", "знак"]
                ):
                    return stripped

        return None

    def parse(
        self, pdf_path: str, ocr_required: bool = False, max_pages: int = None
    ) -> List["ChunkWithCitation"]:
        if ocr_required:
            logger.warning(
                f"TextPDFParser called for {pdf_path}, but OCR is required. Use OCRPDFParser instead."
            )
            return []

        try:
            title = self._extract_title_from_path(pdf_path)
            chunks = []

            doc = self.fitz.open(pdf_path)
            total_pages = len(doc)

            # Limit pages if specified
            if max_pages and total_pages > max_pages:
                logger.warning(f"PDF has {total_pages} pages, limiting to first {max_pages} pages")
                pages_to_process = max_pages
            else:
                pages_to_process = total_pages

            logger.info(f"Processing text PDF: {pdf_path} ({pages_to_process}/{total_pages} pages)")

            all_pages_text = []
            for page_num in range(pages_to_process):
                page = doc[page_num]
                # get_text("text") preserves spaces better than pdfplumber
                # Also try "blocks" mode for structured extraction
                page_text = page.get_text("text") or ""
                all_pages_text.append(page_text)

                if (page_num + 1) % 50 == 0:
                    logger.info(f"Extracted text from {page_num + 1}/{pages_to_process} pages")

            doc.close()

            for page_num, page_text in enumerate(all_pages_text, 1):
                if not page_text.strip():
                    continue

                section = self._detect_section(page_text, page_num, all_pages_text)

                from .semantic_chunker import SemanticChunkConfig, chunk_text_semantic

                page_chunks = chunk_text_semantic(
                    text=page_text,
                    source_file=os.path.basename(pdf_path),
                    doc_title=title,
                    page_num=page_num,
                    section=section,
                    config=SemanticChunkConfig(),
                )
                chunks.extend(page_chunks)

            logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
            return chunks

        except Exception as e:
            logger.error(f"Error parsing text PDF {pdf_path}: {e}")
            raise
