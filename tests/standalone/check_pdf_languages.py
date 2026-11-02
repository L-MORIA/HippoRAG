import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hipporag.pdf_ingestion.pdf_processor import PDFProcessor

# Check all PDF files
pdf_dir = Path("data/pdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files:")
for pdf_file in pdf_files:
    print(f"\n{'='*80}")
    print(f"File: {pdf_file.name}")
    print(f"Size: {pdf_file.stat().st_size / 1024:.2f} KB")

    # Try to extract first page to check language
    try:
        processor = PDFProcessor()
        chunks = processor.process_pdf(str(pdf_file), chunk_size=500, chunk_overlap=50)

        if chunks:
            first_chunk = chunks[0]
            print(f"\nTotal chunks: {len(chunks)}")
            print(f"\nFirst chunk preview (200 chars):")
            print(first_chunk[:200])

            # Check if English
            has_english = any(
                word in first_chunk.lower()
                for word in [" the ", " and ", " is ", " are ", " this ", " that "]
            )
            has_cyrillic = any(
                char in first_chunk
                for char in "абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            )

            if has_english and not has_cyrillic:
                print("\n⚠️  ENGLISH DOCUMENT DETECTED!")
            elif has_cyrillic:
                print("\n✓ Russian document")
            else:
                print("\n? Unknown language")
    except Exception as e:
        print(f"Error processing: {e}")
