#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрая перестройка графа с улучшенной фильтрацией entities.
Не перепарсит PDF и не переэмбеддит чанки.
Время: ~10-15 минут.
"""

import os, sys, json, logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SAVE_DIR = "outputs/astrology_test"
LLM_NAME = os.getenv("LLM_NAME", "gemma-4-e4b")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import normalize_entity_name, is_valid_entity, extract_entity_nodes, text_processing
from hipporag.information_extraction.ner_russian import RussianNER


def main():
    logger.info("=" * 60)
    logger.info("ПЕРЕСТРОЙКА ГРАФА: улучшенная фильтрация entities")
    logger.info("=" * 60)

    config = BaseConfig(
        save_dir=SAVE_DIR,
        llm_name=LLM_NAME,
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        embedding_model_name=EMBEDDING_MODEL,
        linking_top_k=100, retrieval_top_k=200,
        passage_node_weight=0.05, embedding_batch_size=8,
    )
    config.force_openie_from_scratch = True
    config.force_index_from_scratch = True

    working_dir = Path(SAVE_DIR) / f"{LLM_NAME}_{EMBEDDING_MODEL.replace('/', '_')}"

    # Удаляем старые entity/fact/граф (чанки оставляем!)
    for subdir in ["entity_embeddings", "fact_embeddings"]:
        d = working_dir / subdir
        if d.exists():
            for f in d.glob("*.parquet"):
                f.unlink()
    graph_path = working_dir / "graph.pickle"
    if graph_path.exists():
        bak = str(graph_path) + ".bak"
        import shutil
        shutil.copy2(str(graph_path), bak)
        graph_path.unlink()
        logger.info(f"Старый граф сохранён как {bak}")

    openie_path = Path(SAVE_DIR) / f"openie_results_ner_{LLM_NAME}.json"
    if openie_path.exists():
        openie_path.unlink()

    # Загружаем HippoRAG
    hipporag = HippoRAG(global_config=config)
    chunk_to_rows = hipporag.chunk_embedding_store.get_all_id_to_rows()
    chunk_ids = list(chunk_to_rows.keys())
    logger.info(f"Загружено чанков: {len(chunk_ids)}")

    # Улучшенный Russian NER
    ner = RussianNER()
    chunk_entities = {}
    logger.info("Извлечение сущностей (улучшенная фильтрация)...")
    for chunk_id in chunk_ids:
        row = chunk_to_rows[chunk_id]
        text = row["content"] if isinstance(row, dict) else row
        entities = ner.extract(text)
        entities = [
            normalize_entity_name(e) for e in entities
            if is_valid_entity(normalize_entity_name(e)) and len(normalize_entity_name(e)) >= 4
        ]
        chunk_entities[chunk_id] = list(dict.fromkeys(entities))

    total = sum(len(v) for v in chunk_entities.values())
    unique = set()
    for v in chunk_entities.values():
        unique.update(v)
    logger.info(f"Извлечено: {total} всего, {len(unique)} уникальных")

    # Минимальные триплеты (без co-occurrence, только для entity extraction)
    chunk_triples = []
    for chunk_id in chunk_ids:
        entities = chunk_entities.get(chunk_id, [])
        triples = []
        for e in entities:
            triples.append([e, "тема", "концепт"])
        chunk_triples.append(triples)

    processed_triples = [
        [text_processing(t) for t in chunk_triples[i]]
        for i, chunk_id in enumerate(chunk_ids)
    ]

    entity_nodes, chunk_triple_entities = extract_entity_nodes(processed_triples)
    logger.info(f"Entity nodes: {len(entity_nodes)}")

    # Entity embeddings
    logger.info("Создание entity embeddings...")
    hipporag.entity_embedding_store.insert_strings(entity_nodes)

    # Placeholder facts
    hipporag.fact_embedding_store.insert_strings(["placeholder"])

    # Граф
    logger.info("Построение графа...")
    hipporag.node_to_node_stats = {}
    hipporag.ent_node_to_chunk_ids = {}
    hipporag.add_fact_edges(chunk_ids, processed_triples)
    num_new = hipporag.add_passage_edges(chunk_ids, chunk_triple_entities)
    if num_new > 0:
        hipporag.add_synonymy_edges()
        hipporag.augment_graph()
        hipporag.save_igraph()
        logger.info(f"Граф: {hipporag.graph.vcount()} узлов, {hipporag.graph.ecount()} рёбер")

    # OpenIE
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

    logger.info("=" * 60)
    logger.info("ПЕРЕСТРОЙКА ЗАВЕРШЕНА!")
    logger.info("=" * 60)
    logger.info(f"Чанков: {len(chunk_ids)}")
    logger.info(f"Entity nodes: {len(entity_nodes)}")
    logger.info(f"Граф: {hipporag.graph.vcount()} узлов, {hipporag.graph.ecount()} рёбер")


if __name__ == "__main__":
    main()
