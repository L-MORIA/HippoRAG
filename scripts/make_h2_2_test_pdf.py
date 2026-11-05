#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генерирует минимальный 1-страничный PDF для живой приёмки H2-2.

Текст ASCII (Helvetica), чтобы извлечение было надёжным; уникальный токен
XYZZY позволяет найти документ после загрузки. Скрипт служебный, в package
data не входит.
"""

from pathlib import Path

OUT = Path("data/acceptance/h2_2_acceptance.pdf")

TEXT = (
    "Acceptance test document for H2-2 data lifecycle. Unique token: XYZZY. "
    "This document verifies the full index lifecycle: upload, reindex, backup and restore. "
    "Astrology practice distinguishes natal chart houses, planets and zodiac signs. "
    "Every planet rules specific zodiac signs and houses. "
    "Mars rules Aries and Scorpio in traditional astrology. "
    "Venus rules Taurus and Libra. Mercury rules Gemini and Virgo. "
    "The Moon rules Cancer and the Sun rules Leo. "
    "Jupiter rules Sagittarius and Pisces. Saturn rules Capricorn and Aquarius. "
    "In feng shui practice each apartment has zones mapped to compass directions. "
    "The north zone is career, the south zone is fame. "
    "The east zone is family and health, the west zone is creativity and children. "
    "The southeast zone is wealth, the southwest zone is partnership. "
    "The northwest zone is mentors and travel, the northeast zone is knowledge. "
    "The center zone is personal self esteem. "
    "A bagua map overlays these zones on the apartment floor plan. "
    "Each zone has element, color and shape associations. "
    "The career zone element is water and its color is black or dark blue. "
    "The fame zone element is fire and its color is red. "
    "The wealth zone element is wood and its color is purple or green. "
    "Mirrors, plants, crystals and fountains are common feng shui remedies. "
    "A mirror in the wealth zone doubles abundance according to tradition. "
    "A healthy green plant in the east zone supports family harmony. "
    "Clutter blocking any zone blocks the corresponding life area. "
    "Regular cleaning of the entrance supports career opportunities. "
    "The kitchen represents nourishment and prosperity in the house. "
    "The bedroom represents rest and relationships. "
    "Keep the stove clean and working, it symbolizes wealth. "
    "Avoid mirrors facing the bed, they disturb rest. "
    "End of acceptance test document for H2-2. Token XYZZY confirmed."
)


def build_pdf(text: str) -> bytes:
    # Разбиваем текст на строки и рисуем каждую отдельным Tj,
    # чтобы fitz надёжно извлекал текст (как из обычной страницы книги).
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > 88:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)

    parts = ["BT /F1 11 Tf 72 740 Td"]
    for i, line in enumerate(lines):
        safe = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        if i == 0:
            parts.append(f"({safe}) Tj")
        else:
            parts.append(f"0 -16 Td ({safe}) Tj")
    parts.append("ET")
    stream = " ".join(parts).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(build_pdf(TEXT))
    import fitz  # тот же парсер, что в pdf_ingestion

    doc = fitz.open(OUT)
    extracted = " ".join(page.get_text().strip() for page in doc)
    doc.close()
    assert "XYZZY" in extracted, f"текст не извлёкся: {extracted!r}"
    print(f"OK: {OUT} ({OUT.stat().st_size} байт), извлечено: {extracted[:80]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
