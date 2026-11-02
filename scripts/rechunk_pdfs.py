#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Переразбивка PDF на чанки с overlap=0 (убрать обрывы фраз).
+ полная перестройка графа (NER, entities, facts).
Время: ~15-20 минут.
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
from hipporag.pdf_ingestion import TextPDFParser
from hipporag.pdf_ingestion.semantic_chunker import SemanticChunkConfig


def main():
    logger.info("=" * 60)
    logger.info("ПЕРЕЧАНКИНГ: overlap=0 + перестройка графа")
    logger.info("=" * 60)

    # 1. Находим PDF
    pdf_dir = Path("data/pdfs")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"Найдено PDF: {len(pdf_files)}")
    for p in pdf_files:
        logger.info(f"  {p.name}")

    # 2. Парсим PDF с overlap=0
    config = SemanticChunkConfig(chunk_overlap_tokens=0)
    all_chunks = []

    for pdf_path in pdf_files:
        logger.info(f"Парсинг: {pdf_path.name}")
        try:
            parser = TextPDFParser()
            chunks = parser.parse(str(pdf_path))
            logger.info(f"  Получено {len(chunks)} чанков (overlap=0)")
            all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"  Ошибка парсинга: {e}")

    logger.info(f"Всего чанков: {len(all_chunks)}")

    # 3. Очищаем старые данные
    working_dir = Path(SAVE_DIR) / f"{LLM_NAME}_{EMBEDDING_MODEL.replace('/', '_')}"
    
    # Удаляем старые чанки
    chunk_dir = working_dir / "chunk_embeddings"
    for f in chunk_dir.glob("*.parquet"):
        f.unlink()
    
    # Удаляем старые entity/fact/граф
    for subdir in ["entity_embeddings", "fact_embeddings"]:
        d = working_dir / subdir
        if d.exists():
            for f in d.glob("*.parquet"):
                f.unlink()
    
    graph_path = working_dir / "graph.pickle"
    if graph_path.exists():
        graph_path.unlink()

    openie_path = Path(SAVE_DIR) / f"openie_results_ner_{LLM_NAME}.json"
    if openie_path.exists():
        openie_path.unlink()

    # 4. Инициализируем HippoRAG (с флагами пересоздания)
    hconfig = BaseConfig(
        save_dir=SAVE_DIR,
        llm_name=LLM_NAME,
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        embedding_model_name=EMBEDDING_MODEL,
        linking_top_k=100, retrieval_top_k=200,
        passage_node_weight=0.05, embedding_batch_size=8,
    )
    hconfig.force_openie_from_scratch = True
    hconfig.force_index_from_scratch = True

    hipporag = HippoRAG(global_config=hconfig)

    # 5. Сохраняем новые чанки
    chunk_texts = [c.text for c in all_chunks]
    chunk_metadatas = [
        {
            "source_file": c.source_file,
            "doc_title": c.doc_title,
            "page_num": c.page_num,
            "section": c.section or "",
            "content": c.text,
        }
        for c in all_chunks
    ]

    logger.info(f"Сохранение {len(chunk_texts)} чанков в embedding store...")
    hipporag.chunk_embedding_store.insert_strings(chunk_texts, chunk_metadatas)
    logger.info(f"Чанки сохранены: {len(hipporag.chunk_embedding_store.embeddings)}")

    # 6. Получаем все чанки
    chunk_to_rows = hipporag.chunk_embedding_store.get_all_id_to_rows()
    chunk_ids = list(chunk_to_rows.keys())
    logger.info(f"Чанков в хранилище: {len(chunk_ids)}")

    # 7. Русский NER (spaCy + словарь, быстро)
    from hipporag.information_extraction.ner_russian import RussianNER
    from hipporag.utils.misc_utils import normalize_entity_name, is_valid_entity, extract_entity_nodes, flatten_facts, text_processing

    russian_ner = RussianNER()
    chunk_entities = {}

    logger.info("Извлечение сущностей русским NER...")
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

    # 8. Создаём минимальные триплеты (только для entity extraction, без co-occurrence)
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
    # Минимум facts: только unique triples (без co-occurrence их мало)
    facts = flatten_facts(processed_triples)

    logger.info(f"Entity nodes: {len(entity_nodes)}")
    logger.info(f"Facts: {len(facts)}")

    # 9. Сохраняем entity embeddings (без fact embeddings — они не нужны для PPR)
    logger.info("Создание entity embeddings...")
    hipporag.entity_embedding_store.insert_strings(entity_nodes)

    logger.info("Пропуск fact embeddings (не нужны)...")
    hipporag.fact_embedding_store.insert_strings(["placeholder"])

    # 10. Строим граф
    logger.info("Построение графа...")
    hipporag.node_to_node_stats = {}
    hipporag.ent_node_to_chunk_ids = {}
    hipporag.add_fact_edges(chunk_ids, processed_triples)
    num_new = hipporag.add_passage_edges(chunk_ids, chunk_triple_entities)

    if num_new > 0:
        logger.info(f"Новых чанков в графе: {num_new}")
        hipporag.add_synonymy_edges()
        hipporag.augment_graph()
        hipporag.save_igraph()
        logger.info(f"Граф: {hipporag.graph.vcount()} узлов, {hipporag.graph.ecount()} рёбер")

    # 11. Сохраняем OpenIE
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

    # 12. Итог
    logger.info("=" * 60)
    logger.info("ПЕРЕЧАНКИНГ ЗАВЕРШЁН!")
    logger.info("=" * 60)
    logger.info(f"Чанков: {len(chunk_ids)}")
    logger.info("Размер чанков: overlap=0 (было overlap=64)")
    logger.info(f"Entity nodes: {len(entity_nodes)}")
    logger.info(f"Граф: {hipporag.graph.vcount()} узлов, {hipporag.graph.ecount()} рёбер")


if __name__ == "__main__":
    main()
