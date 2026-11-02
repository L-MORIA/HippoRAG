from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional

from .base import ChunkWithCitation

if TYPE_CHECKING:
    # PIL нужен только в рантайме (импорт в __init__ с graceful-ошибкой);
    # здесь — исключительно для аннотаций
    from PIL import Image

logger = logging.getLogger(__name__)


class OCRConfig:
    def __init__(self, enabled=True, language="eng", dpi=300, engine_mode=1):
        self.enabled = enabled
        self.language = language
        self.dpi = dpi
        self.engine_mode = engine_mode


class OCRPDFParser:

    def __init__(self, ocr_config=None):
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError(
                "pytesseract and Pillow not installed. Install with: pip install pytesseract Pillow"
            )

        self.pytesseract = pytesseract
        self.Image = Image

        try:
            import fitz

            self.fitz = fitz
        except ImportError:
            raise ImportError("PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")
        self.fitz = fitz

        if ocr_config is None:
            self.ocr_config = OCRConfig()
        else:
            if isinstance(ocr_config, dict):
                self.ocr_config = OCRConfig(
                    enabled=ocr_config.get("enabled", True),
                    language=ocr_config.get("language", "eng"),
                    dpi=ocr_config.get("dpi", 300),
                    engine_mode=ocr_config.get("engine_mode", 1),
                )
            else:
                self.ocr_config = ocr_config

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

    def _is_text_pdf(self, pdf_path: str) -> bool:
        doc = self.fitz.open(pdf_path)
        first_page_text = doc[0].get_text("text")
        doc.close()

        text_ratio = len(first_page_text.strip()) / max(len(first_page_text), 1)
        return text_ratio > 0.1

    def _extract_section_from_ocr(self, ocr_text: str) -> Optional[str]:
        lines = ocr_text.strip().split("\n")
        for line in lines[:5]:
            stripped = line.strip()
            if len(stripped) > 10 and len(stripped) < 100:
                if any(
                    keyword in stripped.lower()
                    for keyword in ["chapter", "глав", "раздел", "section", "part", "часть"]
                ):
                    return stripped
        return None

    def _page_to_image(self, page, dpi=300) -> "Image.Image":
        mat = self.fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat)
        img = self.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    def _ocr_page(self, image: "Image.Image") -> str:
        text = self.pytesseract.image_to_string(image, lang=self.ocr_config.language)
        return text

    def parse(self, pdf_path: str, force_ocr: bool = False) -> List["ChunkWithCitation"]:

        try:
            title = self._extract_title_from_path(pdf_path)

            if not force_ocr and self._is_text_pdf(pdf_path):
                logger.info(f"PDF {pdf_path} appears to be text-based, using TextPDFParser")
                from .text_pdf_parser import TextPDFParser

                parser = TextPDFParser()
                return parser.parse(pdf_path)

            if not self.ocr_config.enabled:
                logger.warning(f"OCR is disabled for {pdf_path}")
                return []

            chunks = []
            doc = self.fitz.open(pdf_path)
            total_pages = len(doc)

            logger.info(f"Processing OCR PDF: {pdf_path} ({total_pages} pages)")

            for page_num in range(total_pages):
                page = doc[page_num]

                text_on_page = page.get_text("text").strip()
                if text_on_page and len(text_on_page) > 50 and not force_ocr:
                    from .chunker import chunk_text_with_metadata

                    section = self._extract_section_from_ocr(text_on_page)
                    page_chunks = chunk_text_with_metadata(
                        text=text_on_page,
                        source_file=os.path.basename(pdf_path),
                        doc_title=title,
                        page_num=page_num + 1,
                        section=section,
                        config=None,
                    )
                    chunks.extend(page_chunks)
                    continue

                image = self._page_to_image(page, dpi=self.ocr_config.dpi)
                ocr_text = self._ocr_page(image)

                if not ocr_text.strip():
                    logger.debug(f"Page {page_num + 1}: No text found by OCR")
                    continue

                from .chunker import chunk_text_with_metadata

                section = self._extract_section_from_ocr(ocr_text)
                page_chunks = chunk_text_with_metadata(
                    text=ocr_text,
                    source_file=os.path.basename(pdf_path),
                    doc_title=title,
                    page_num=page_num + 1,
                    section=section,
                    config=None,
                )
                chunks.extend(page_chunks)

            doc.close()
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_path} (OCR)")
            return chunks

        except Exception as e:
            logger.error(f"Error parsing OCR PDF {pdf_path}: {e}")
            raise
