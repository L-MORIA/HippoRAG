#!/usr/bin/env python3
"""
Показать статистику индекса HippoRAG.
"""
import os
import sys
from collections import Counter
import argparse

import pandas as pd
import numpy as np
import pickle
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hipporag.utils.config_utils import build_working_dir

# Parse command line arguments
parser = argparse.ArgumentParser(description='Show HippoRAG index statistics')
parser.add_argument('--save_dir', default='outputs/astrology', help='Output directory')
args = parser.parse_args()

SAVE_DIR = args.save_dir
LLM_NAME = os.getenv('LLM_NAME', 'gemma-4-e4b')
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

working_dir = build_working_dir(SAVE_DIR, LLM_NAME, EMBEDDING_MODEL)

print("=" * 80)
print("INDEX STATISTICS")
print("=" * 80)
print()

# Chunk embeddings
chunk_path = os.path.join(working_dir, 'chunk_embeddings', 'vdb_chunk.parquet')
if os.path.exists(chunk_path):
    df_chunks = pd.read_parquet(chunk_path)
    print(f"📄 CHUNKS: {len(df_chunks)}")
    
    # Chunk sizes
    chunk_lengths = df_chunks['content'].str.len()
    print("   Размер чанков (символы):")
    print(f"     Min:    {chunk_lengths.min()}")
    print(f"     Max:    {chunk_lengths.max()}")
    print(f"     Mean:   {chunk_lengths.mean():.1f}")
    print(f"     Median: {chunk_lengths.median():.1f}")
    
    # Sources
    import json
    sources = []
    for m in df_chunks['metadata']:
        try:
            meta = json.loads(m) if isinstance(m, str) else m
            sources.append(meta.get('source_file', 'unknown'))
        except:
            sources.append('unknown')
    source_counts = Counter(sources)
    print("\n   По источникам:")
    for source, count in source_counts.most_common():
        print(f"     {source}: {count} чанков")
else:
    print("❌ Chunk embeddings не найдены")

print()

# Entity embeddings
entity_path = os.path.join(working_dir, 'entity_embeddings', 'vdb_entity.parquet')
if os.path.exists(entity_path):
    df_entities = pd.read_parquet(entity_path)
    print(f"🏷️  ENTITIES: {len(df_entities)}")
    
    # Top entities by frequency (assuming hash_id contains entity name)
    entity_names = df_entities['content'].tolist()
    entity_counts = Counter(entity_names)
    print("\n   Топ-15 самых частых сущностей:")
    for entity, count in entity_counts.most_common(15):
        print(f"     {entity[:50]:<50} ({count})")
else:
    print("❌ Entity embeddings не найдены")

print()

# Fact embeddings
fact_path = os.path.join(working_dir, 'fact_embeddings', 'vdb_fact.parquet')
if os.path.exists(fact_path):
    df_facts = pd.read_parquet(fact_path)
    print(f"🔗 FACTS (triples): {len(df_facts)}")
else:
    print("❌ Fact embeddings не найдены")

print()

# Graph
graph_path = os.path.join(working_dir, 'graph.pickle')
if os.path.exists(graph_path):
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    print("🕸️  GRAPH:")
    print(f"   Узлов:  {graph.vcount()}")
    print(f"   Рёбер:  {graph.ecount()}")
    
    # Degree distribution
    degrees = graph.degree()
    print("\n   Распределение степеней узлов:")
    print(f"     Min:    {min(degrees)}")
    print(f"     Max:    {max(degrees)}")
    print(f"     Mean:   {np.mean(degrees):.1f}")
    print(f"     Median: {np.median(degrees):.1f}")
    
    # Orphan nodes (degree = 1)
    orphans = sum(1 for d in degrees if d == 1)
    print(f"\n   Orphan nodes (degree=1): {orphans} ({orphans/len(degrees)*100:.1f}%)")
    
    # Hub nodes (degree > 50)
    hubs = sum(1 for d in degrees if d > 50)
    print(f"   Hub nodes (degree>50):   {hubs}")
    
    # Top nodes by degree
    node_degrees = [(i, d) for i, d in enumerate(degrees)]
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    print("\n   Топ-10 узлов по степени:")
    for i, (node_id, degree) in enumerate(node_degrees[:10], 1):
        node_name = graph.vs[node_id]['name'] if 'name' in graph.vs.attributes() else f"node_{node_id}"
        print(f"     {i:2}. {node_name[:50]:<50} (degree={degree})")
    
    # Edge types (if available)
    if 'type' in graph.es.attributes():
        edge_types = Counter(graph.es['type'])
        print("\n   Типы рёбер:")
        for etype, count in edge_types.most_common():
            print(f"     {etype}: {count}")
else:
    print("❌ Graph не найден")

print()
print("=" * 80)
