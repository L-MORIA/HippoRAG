#!/usr/bin/env python3
"""
Очистка графа от мусорных entities (пустые строки, числа, короткие слова).
"""
import os
import sys
import pickle
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hipporag.utils.config_utils import build_working_dir

SAVE_DIR = 'outputs/astrology'
LLM_NAME = os.getenv('LLM_NAME', 'gemma-4-e4b')
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

working_dir = build_working_dir(SAVE_DIR, LLM_NAME, EMBEDDING_MODEL)

print("=" * 80)
print("GRAPH CLEANING")
print("=" * 80)
print()

# Load graph
graph_path = os.path.join(working_dir, 'graph.pickle')
if not os.path.exists(graph_path):
    print(f"❌ Graph not found at {graph_path}")
    sys.exit(1)

print(f"📂 Loading graph from {graph_path}")
with open(graph_path, 'rb') as f:
    graph = pickle.load(f)

print(f"   Original: {graph.vcount()} nodes, {graph.ecount()} edges")
print()

# Identify garbage nodes
print("🔍 Identifying garbage nodes...")

garbage_node_ids = []
garbage_reasons = {}

for node_id in range(graph.vcount()):
    node_name = graph.vs[node_id]['name']
    
    # Extract entity name from node name (format: "entity-<hash>")
    if not node_name.startswith('entity-'):
        continue  # Skip passage nodes
    
    # Get actual entity content
    # We need to look up the entity by hash in entity_embeddings
    entity_hash = node_name.replace('entity-', '')
    
    # For now, check node degree - the empty entity has degree 15696
    degree = graph.degree(node_id)
    
    # Flag suspicious nodes:
    # 1. Extremely high degree (likely empty entity)
    if degree > 1000:
        garbage_node_ids.append(node_id)
        garbage_reasons[node_id] = f"extreme_degree_{degree}"
        continue

print(f"   Found {len(garbage_node_ids)} garbage nodes by degree analysis")
print()

# Load entity embeddings to check content
entity_path = os.path.join(working_dir, 'entity_embeddings', 'vdb_entity.parquet')
if os.path.exists(entity_path):
    print("📄 Loading entity embeddings to check content...")
    df_entities = pd.read_parquet(entity_path)
    
    # Create hash to content mapping
    hash_to_content = {}
    for idx, row in df_entities.iterrows():
        # hash_id is the MD5 hash of the entity
        entity_hash = row['hash_id']
        entity_content = row['content']
        hash_to_content[entity_hash] = entity_content
    
    print(f"   Loaded {len(hash_to_content)} entity mappings")
    
    # Now check all entity nodes
    for node_id in range(graph.vcount()):
        node_name = graph.vs[node_id]['name']
        
        if not node_name.startswith('entity-'):
            continue
        
        entity_hash = node_name.replace('entity-', '')
        
        if entity_hash in hash_to_content:
            content = hash_to_content[entity_hash]
            
            # Filter criteria:
            # 1. Empty string
            # 2. Length < 3
            # 3. Pure digits
            if not content or len(content) < 3 or content.isdigit():
                if node_id not in garbage_node_ids:
                    garbage_node_ids.append(node_id)
                    garbage_reasons[node_id] = f"bad_content:'{content}'"
    
    print(f"   Total garbage nodes: {len(garbage_node_ids)}")
    print()

# Show top garbage nodes
print("🗑️  Top 20 garbage nodes:")
garbage_with_degree = [(nid, graph.degree(nid), garbage_reasons.get(nid, 'unknown')) 
                       for nid in garbage_node_ids]
garbage_with_degree.sort(key=lambda x: x[1], reverse=True)

for i, (node_id, degree, reason) in enumerate(garbage_with_degree[:20], 1):
    node_name = graph.vs[node_id]['name']
    print(f"   {i:2}. {node_name[:50]:<50} degree={degree:5} reason={reason}")

print()

# Ask for confirmation
print(f"⚠️  About to delete {len(garbage_node_ids)} nodes")
response = input("Continue? (yes/no): ")

if response.lower() != 'yes':
    print("❌ Aborted")
    sys.exit(0)

print()
print("🧹 Deleting garbage nodes...")

# Delete nodes (delete in reverse order to avoid index shifting)
garbage_node_ids.sort(reverse=True)
graph.delete_vertices(garbage_node_ids)

print(f"   ✅ Deleted {len(garbage_node_ids)} nodes")
print(f"   New graph: {graph.vcount()} nodes, {graph.ecount()} edges")
print()

# Save cleaned graph
backup_path = graph_path + '.backup'
print(f"💾 Backing up original graph to {backup_path}")
os.rename(graph_path, backup_path)

print(f"💾 Saving cleaned graph to {graph_path}")
with open(graph_path, 'wb') as f:
    pickle.dump(graph, f)

print()
print("=" * 80)
print("✅ GRAPH CLEANING COMPLETED")
print("=" * 80)
print()
print(f"Original: {graph.vcount() + len(garbage_node_ids)} nodes")
print(f"Cleaned:  {graph.vcount()} nodes")
print(f"Removed:  {len(garbage_node_ids)} nodes ({len(garbage_node_ids)/(graph.vcount() + len(garbage_node_ids))*100:.1f}%)")
print()
print(f"Backup saved to: {backup_path}")
