import json

import pandas as pd

# Load the chunk embeddings
df = pd.read_parquet(
    "outputs/astrology_full/gemma-4-e4b_sentence-transformers_all-MiniLM-L6-v2/chunk_embeddings/vdb_chunk.parquet"
)

print(f"Total chunks: {len(df)}")
print("\nColumns:", df.columns.tolist())

# Parse metadata and find all unique source files
print("\n=== All unique source files ===")
metas = []
for _, row in df.iterrows():
    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    metas.append(meta)

sources = {}
for i, meta in enumerate(metas):
    source = meta.get("source_file", "unknown")
    if source not in sources:
        sources[source] = []
    sources[source].append(i)

for source, indices in sources.items():
    print(f"  - {source}: {len(indices)} chunks")

# Look for English content
print("\n=== Searching for English content ===")
english_chunks = []
for idx, row in df.iterrows():
    meta = metas[idx]
    content = row["content"]
    content_preview = content[:200]

    # Check if content looks like English (has common English words and no Cyrillic)
    has_english_words = any(
        word in content_preview.lower()
        for word in [" the ", " and ", " is ", " are ", " this ", " that ", " with ", " from "]
    )
    has_cyrillic = any(
        char in content
        for char in "абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    )

    if has_english_words and not has_cyrillic:
        english_chunks.append(
            {
                "index": idx,
                "hash_id": row["hash_id"],
                "source": meta.get("source_file", "unknown"),
                "content_preview": content_preview,
            }
        )

print(f"\nFound {len(english_chunks)} potential English chunks:")
for chunk in english_chunks:
    print(f"\nIndex {chunk['index']}: {chunk['source']}")
    print(f"Hash ID: {chunk['hash_id']}")
    print(f"Preview: {chunk['content_preview']}")
    print("-" * 80)
