import json
from pathlib import Path

import pandas as pd

# Find all parquet files
parquet_files = list(Path("outputs").rglob("vdb_chunk.parquet"))

print(f"Found {len(parquet_files)} chunk parquet files:\n")

all_english_chunks = []

for parquet_file in parquet_files:
    print(f"{'='*80}")
    print(f"File: {parquet_file}")

    try:
        df = pd.read_parquet(parquet_file)
        print(f"Total chunks: {len(df)}")

        # Parse metadata and find sources
        sources = {}
        for idx, row in df.iterrows():
            meta = (
                json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            )
            source = meta.get("source_file", "unknown")
            if source not in sources:
                sources[source] = []
            sources[source].append(idx)

        print(f"Sources:")
        for source, indices in sources.items():
            print(f"  - {source}: {len(indices)} chunks")

        # Check for English content
        english_count = 0
        for idx, row in df.iterrows():
            meta = (
                json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            )
            content = row["content"]

            # Check if English
            has_english = any(
                word in content.lower()
                for word in [
                    " the ",
                    " and ",
                    " is ",
                    " are ",
                    " this ",
                    " that ",
                    " with ",
                    " from ",
                ]
            )
            has_cyrillic = any(
                char in content
                for char in "абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            )

            if has_english and not has_cyrillic:
                english_count += 1
                all_english_chunks.append(
                    {
                        "file": str(parquet_file),
                        "index": idx,
                        "hash_id": row["hash_id"],
                        "source": meta.get("source_file", "unknown"),
                        "content_preview": content[:150],
                    }
                )

        if english_count > 0:
            print(f"⚠️  Found {english_count} English chunks!")

    except Exception as e:
        print(f"Error: {e}")

    print()

print(f"\n{'='*80}")
print(f"SUMMARY: Found {len(all_english_chunks)} total English chunks across all files")

if all_english_chunks:
    print("\nEnglish chunks details:")
    for i, chunk in enumerate(all_english_chunks, 1):
        print(f"\n{i}. File: {chunk['file']}")
        print(f"   Index: {chunk['index']}, Hash: {chunk['hash_id']}")
        print(f"   Source: {chunk['source']}")
        print(f"   Preview: {chunk['content_preview']}")
