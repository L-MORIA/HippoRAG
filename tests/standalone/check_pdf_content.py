from pathlib import Path

import PyPDF2

pdf_dir = Path("data/pdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files:\n")

for pdf_file in pdf_files:
    print(f"{'='*80}")
    print(f"File: {pdf_file.name}")
    print(f"Size: {pdf_file.stat().st_size / 1024:.2f} KB")

    try:
        with open(pdf_file, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            print(f"Pages: {num_pages}")

            # Extract first page text
            if num_pages > 0:
                first_page = pdf_reader.pages[0]
                text = first_page.extract_text()

                # Show first 300 characters
                preview = text[:300].strip()
                print(f"\nFirst page preview:")
                print(preview)

                # Check language
                has_english = any(
                    word in text.lower()
                    for word in [" the ", " and ", " is ", " are ", " this ", " that ", " with "]
                )
                has_cyrillic = any(
                    char in text
                    for char in "абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
                )

                if has_english and not has_cyrillic:
                    print("\n⚠️  ENGLISH DOCUMENT DETECTED!")
                elif has_cyrillic:
                    print("\n✓ Russian document")
                else:
                    print("\n? Unknown language")

    except Exception as e:
        print(f"Error: {e}")

    print()
