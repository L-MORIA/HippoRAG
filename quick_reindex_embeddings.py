"""
Быстрая переиндексация: обновляем только embeddings с новой моделью
Используем существующий граф знаний и документы
"""
import os
import sys
import pandas as pd
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 80)
print("БЫСТРАЯ ПЕРЕИНДЕКСАЦИЯ EMBEDDINGS")
print("=" * 80)

# Paths
old_dir = Path("outputs/astrology/gemma-4-e4b_sentence-transformers_all-MiniLM-L6-v2")
new_dir = Path("outputs/astrology/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2")

print(f"\nИсходная директория: {old_dir}")
print(f"Новая директория: {new_dir}")

# Create new directory
new_dir.mkdir(parents=True, exist_ok=True)

# Load multilingual model
print("\nЗагрузка multilingual embedding модели...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
print("✓ Модель загружена")

# Copy graph (no need to rebuild)
print("\nКопирование графа знаний...")
graph_src = old_dir / "graph.pickle"
graph_dst = new_dir / "graph.pickle"
if graph_src.exists():
    import shutil
    shutil.copy2(graph_src, graph_dst)
    print(f"✓ Граф скопирован: {graph_dst}")

# Re-encode chunks
print("\n" + "=" * 80)
print("ПЕРЕИНДЕКСАЦИЯ CHUNK EMBEDDINGS")
print("=" * 80)

chunk_dir_old = old_dir / "chunk_embeddings"
chunk_dir_new = new_dir / "chunk_embeddings"
chunk_dir_new.mkdir(parents=True, exist_ok=True)

chunk_file = chunk_dir_old / "vdb_chunk.parquet"
if chunk_file.exists():
    print(f"\nЗагрузка chunks из {chunk_file}")
    df = pd.read_parquet(chunk_file)
    print(f"✓ Загружено {len(df)} chunks")
    
    print("\nПереиндексация embeddings...")
    contents = df['content'].tolist()
    
    # Encode in batches
    batch_size = 32
    all_embeddings = []
    for i in tqdm(range(0, len(contents), batch_size), desc="Encoding chunks"):
        batch = contents[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings)
    
    # Update dataframe
    df['embedding'] = all_embeddings
    
    # Save
    output_file = chunk_dir_new / "vdb_chunk.parquet"
    df.to_parquet(output_file)
    print(f"✓ Сохранено в {output_file}")

# Re-encode entities
print("\n" + "=" * 80)
print("ПЕРЕИНДЕКСАЦИЯ ENTITY EMBEDDINGS")
print("=" * 80)

entity_dir_old = old_dir / "entity_embeddings"
entity_dir_new = new_dir / "entity_embeddings"
entity_dir_new.mkdir(parents=True, exist_ok=True)

entity_file = entity_dir_old / "vdb_entity.parquet"
if entity_file.exists():
    print(f"\nЗагрузка entities из {entity_file}")
    df = pd.read_parquet(entity_file)
    print(f"✓ Загружено {len(df)} entities")
    
    print("\nПереиндексация embeddings...")
    contents = df['content'].tolist()
    
    all_embeddings = []
    for i in tqdm(range(0, len(contents), batch_size), desc="Encoding entities"):
        batch = contents[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings)
    
    df['embedding'] = all_embeddings
    
    output_file = entity_dir_new / "vdb_entity.parquet"
    df.to_parquet(output_file)
    print(f"✓ Сохранено в {output_file}")

# Re-encode facts
print("\n" + "=" * 80)
print("ПЕРЕИНДЕКСАЦИЯ FACT EMBEDDINGS")
print("=" * 80)

fact_dir_old = old_dir / "fact_embeddings"
fact_dir_new = new_dir / "fact_embeddings"
fact_dir_new.mkdir(parents=True, exist_ok=True)

fact_file = fact_dir_old / "vdb_fact.parquet"
if fact_file.exists():
    print(f"\nЗагрузка facts из {fact_file}")
    df = pd.read_parquet(fact_file)
    print(f"✓ Загружено {len(df)} facts")
    
    print("\nПереиндексация embeddings...")
    contents = df['content'].tolist()
    
    all_embeddings = []
    for i in tqdm(range(0, len(contents), batch_size), desc="Encoding facts"):
        batch = contents[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings)
    
    df['embedding'] = all_embeddings
    
    output_file = fact_dir_new / "vdb_fact.parquet"
    df.to_parquet(output_file)
    print(f"✓ Сохранено в {output_file}")

print("\n" + "=" * 80)
print("ПЕРЕИНДЕКСАЦИЯ ЗАВЕРШЕНА!")
print("=" * 80)
print(f"\nНовый индекс сохранен в: {new_dir}")
print("\nТеперь можно запустить search_interactive.py")
