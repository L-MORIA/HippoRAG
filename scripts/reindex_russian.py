#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрая переиндексация с русским NER.
Перестраивает entities, facts и граф (БЕЗ пересоздания чанков).
Время работы: ~5-10 минут (вместо 7 часов).

Использует:
  - spaCy ru_core_news_sm для NER
  - Словарь доменных терминов (знаки зодиака, планеты, фэншуй)
  - Co-occurrence триплеты (без LLM)
"""

import os, sys, json, shutil, logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SAVE_DIR = "outputs/astrology_test"
LLM_NAME = os.getenv("LLM_NAME", "gemma-4-e4b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import (
    normalize_entity_name, is_valid_entity,
    extract_entity_nodes, flatten_facts, text_processing,
)
from hipporag.information_extraction.ner_russian import RussianNER


def main():
    logger.info("=" * 60)
    logger.info("ПЕРЕИНДЕКСАЦИЯ: русский NER + перестройка графа")
    logger.info("=" * 60)

    config = BaseConfig(
        save_dir=SAVE_DIR,
        llm_name=LLM_NAME,
        llm_base_url=LLM_BASE_URL,
        embedding_model_name=EMBEDDING_MODEL,
        linking_top_k=100, retrieval_top_k=200,
        passage_node_weight=0.05, embedding_batch_size=8,
    )
    config.force_openie_from_scratch = True
    config.force_index_from_scratch = True

    working_dir = Path(SAVE_DIR) / f"{LLM_NAME}_{EMBEDDING_MODEL.replace('/', '_')}"

    # Backup and remove old files BEFORE initializing HippoRAG
    for path in [
        working_dir / "graph.pickle",
        Path(SAVE_DIR) / f"openie_results_ner_{LLM_NAME}.json",
    ]:
        if path.exists():
            backup = str(path) + ".bak"
            shutil.copy2(str(path), backup)
            path.unlink()
            logger.info(f"Удалён {path.name} (бэкап: {backup})")

    # Remove entity/fact embedding files
    for subdir in ["entity_embeddings", "fact_embeddings"]:
        d = working_dir / subdir
        if d.exists():
            for f in d.glob("*.parquet"):
                f.unlink()

    logger.info("Загрузка HippoRAG (с пустым графом)...")
    hipporag = HippoRAG(global_config=config)
    logger.info(f"Загружено чанков: {len(hipporag.chunk_embedding_store.embeddings)}")

    # Get all chunks
    chunk_to_rows = hipporag.chunk_embedding_store.get_all_id_to_rows()
    chunk_ids = list(chunk_to_rows.keys())
    logger.info(f"Чанков для обработки: {len(chunk_ids)}")

    # Russian NER on all chunks
    russian_ner = RussianNER()
    chunk_entities = {}

    logger.info("Извлечение сущностей русским NER (spaCy + словарь)...")
    for chunk_id in chunk_ids:
        row = chunk_to_rows[chunk_id]
        text = row["content"] if isinstance(row, dict) else row
        entities = russian_ner.extract(text)
        entities = [
            normalize_entity_name(e) for e in entities
            if is_valid_entity(normalize_entity_name(e))
        ]
        chunk_entities[chunk_id] = list(dict.fromkeys(entities))

    total = sum(len(v) for v in chunk_entities.values())
    unique = set()
    for v in chunk_entities.values():
        unique.update(v)
    logger.info(f"Извлечено: {total} всего, {len(unique)} уникальных")

    # Sample entities
    rus_entities = [e for e in unique if any(ord(c) > 0x0400 for c in e)]
    eng_entities = [e for e in unique if all(ord(c) < 0x0400 for c in e)]
    logger.info(f"Русских: {len(rus_entities)}, Английских: {len(eng_entities)}")
    logger.info(f"Примеры русских: {rus_entities[:20]}")

    # Create minimal triples (only for entity extraction, NOT for fact embedding)
    chunk_triples = []
    for chunk_id in chunk_ids:
        entities = chunk_entities.get(chunk_id, [])
        triples = []
        for e in entities:
            triples.append([e, "тема", "концепт"])
        chunk_triples.append(triples)

    # Process triples
    processed_triples = [
        [text_processing(t) for t in chunk_triples[i]]
        for i, chunk_id in enumerate(chunk_ids)
    ]

    # Extract entity nodes and facts
    entity_nodes, chunk_triple_entities = extract_entity_nodes(processed_triples)
    facts = flatten_facts(processed_triples)

    logger.info(f"Entity nodes: {len(entity_nodes)}")
    logger.info(f"Facts: {len(facts)}")

    # Rebuild entity embeddings (stores start empty since we deleted parquet files)
    logger.info("Создание entity embeddings (~5 мин)...")
    hipporag.entity_embedding_store.insert_strings(entity_nodes)
    logger.info(f"Entity embeddings: {len(hipporag.entity_embedding_store.embeddings)}")

    # Skip fact embeddings (not needed for direct entity linking PPR)
    # Insert only minimal placeholder to avoid errors
    logger.info("Пропуск fact embeddings (не нужны для PPR)...")
    hipporag.fact_embedding_store.insert_strings(["placeholder fact"])

    # Build graph
    logger.info("Построение графа...")
    hipporag.node_to_node_stats = {}
    hipporag.ent_node_to_chunk_ids = {}
    hipporag.add_fact_edges(chunk_ids, processed_triples)
    num_new_chunks = hipporag.add_passage_edges(chunk_ids, chunk_triple_entities)

    if num_new_chunks > 0:
        logger.info(f"Новых чанков в графе: {num_new_chunks}")
        hipporag.add_synonymy_edges()
        hipporag.augment_graph()
        hipporag.save_igraph()
        logger.info(f"Граф сохранён: {hipporag.graph.vcount()} узлов, {hipporag.graph.ecount()} рёбер")

    # Save OpenIE results (for compatibility)
    openie_path = Path(SAVE_DIR) / f"openie_results_ner_{LLM_NAME}.json"
    openie_data = {"docs": []}
    for ci, chunk_id in enumerate(chunk_ids):
        row = chunk_to_rows[chunk_id]
        passage = row["content"] if isinstance(row, dict) else row
        openie_data["docs"].append({
            "idx": chunk_id,
            "passage": passage,
            "extracted_entities": chunk_entities.get(chunk_id, []),
            "extracted_triples": processed_triples[ci],
        })
    with open(str(openie_path), "w", encoding="utf-8") as f:
        json.dump(openie_data, f, ensure_ascii=False, indent=2)
    logger.info(f"OpenIE сохранены: {openie_path}")

    logger.info("=" * 60)
    logger.info("ПЕРЕИНДЕКСАЦИЯ ЗАВЕРШЕНА!")
    logger.info("=" * 60)
    logger.info(f"Чанков: {len(chunk_ids)}")
    logger.info(f"Entity nodes: {len(entity_nodes)}")
    logger.info(f"Facts: {len(facts)}")
    logger.info(f"Граф: {hipporag.graph.vcount()} узлов, {hipporag.graph.ecount()} рёбер")
    logger.info(f"Русских entity: {sum(1 for e in entity_nodes if any(ord(c) > 0x0400 for c in e))}")


if __name__ == "__main__":
    main()
