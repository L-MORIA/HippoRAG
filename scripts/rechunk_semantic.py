#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Семантическое переразбиение чанков: группировка предложений по смыслу (embedding similarity).
Текст → предложения → эмбеддинг → группировка по косинусной похожести > 0.7
+ полная перестройка графа.
Время: ~10-15 минут.
"""

import os, sys, json, logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import tiktoken

SAVE_DIR = "outputs/astrology_test"
LLM_NAME = os.getenv("LLM_NAME", "gemma-4-e4b")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SIM_THRESHOLD = 0.7
MAX_TOKENS = 512

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.pdf_ingestion import TextPDFParser
from hipporag.pdf_ingestion.semantic_chunker import clean_pdf_text, split_into_sentences


def main():
    logger.info("=" * 60)
    logger.info("СЕМАНТИЧЕСКОЕ ПЕРЕРАЗБИЕНИЕ ЧАНКОВ")
    logger.info("=" * 60)

    # 1. Загружаем embedding модель
    from sentence_transformers import SentenceTransformer
    logger.info("Загрузка embedding модели...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 2. Парсим PDF
    pdf_dir = Path("data/pdfs")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"Найдено PDF: {len(pdf_files)}")

    all_chunks = []
    encoder = tiktoken.get_encoding("cl100k_base")

    for pdf_path in pdf_files:
        logger.info(f"Парсинг: {pdf_path.name}")
        try:
            parser = TextPDFParser()
            old_chunks = parser.parse(str(pdf_path))
            # Собираем текст по страницам
            page_texts = {}
            for c in old_chunks:
                if c.page_num not in page_texts:
                    page_texts[c.page_num] = []
                page_texts[c.page_num].append(c.text)

            for page_num in sorted(page_texts.keys()):
                page_text = "\n".join(page_texts[page_num])
                page_text = clean_pdf_text(page_text)
                sentences = split_into_sentences(page_text)

                if not sentences:
                    sentences = [s.strip() for s in page_text.split('\n\n') if s.strip()]
                if not sentences:
                    continue

                # Эмбеддинг предложений
                sent_embs = model.encode(sentences)

                # Группировка по сходству
                current_group = [sentences[0]]
                current_emb = sent_embs[0]
                current_tokens = len(encoder.encode(sentences[0]))

                for i in range(1, len(sentences)):
                    sent = sentences[i]
                    sent_tokens = len(encoder.encode(sent))

                    # Похожесть с предыдущим
                    sim = float(np.dot(current_emb, sent_embs[i]))

                    # Если похожесть низкая ИЛИ слишком много токенов → новый чанк
                    if (sim < SIM_THRESHOLD and current_tokens > 50) or current_tokens + sent_tokens > MAX_TOKENS:
                        # Сохраняем текущий чанк
                        chunk_text = " ".join(current_group)
                        if len(chunk_text) > 20:
                            from hipporag.pdf_ingestion.base import ChunkWithCitation
                            first_chunk = [c for c in old_chunks if c.page_num == page_num][0]
                            all_chunks.append(ChunkWithCitation(
                                text=chunk_text,
                                source_file=str(pdf_path.name),
                                doc_title=first_chunk.doc_title,
                                page_num=page_num,
                            ))
                        current_group = [sent]
                        current_emb = sent_embs[i]
                        current_tokens = sent_tokens
                    else:
                        current_group.append(sent)
                        # Обновляем эмбеддинг как среднее
                        current_emb = (current_emb * (len(current_group) - 1) + sent_embs[i]) / len(current_group)
                        current_tokens += sent_tokens

                # Последний чанк
                if current_group:
                    chunk_text = " ".join(current_group)
                    if len(chunk_text) > 20:
                        from hipporag.pdf_ingestion.base import ChunkWithCitation
                        first_chunk = [c for c in old_chunks if c.page_num == page_num][0]
                        all_chunks.append(ChunkWithCitation(
                            text=chunk_text,
                            source_file=str(pdf_path.name),
                            doc_title=first_chunk.doc_title,
                            page_num=page_num,
                        ))

            logger.info(f"  Создано чанков: {len([c for c in all_chunks if c.source_file == str(pdf_path.name)])}")

        except Exception as e:
            logger.warning(f"  Ошибка парсинга {pdf_path.name}: {e}")

    logger.info(f"Всего семантических чанков: {len(all_chunks)}")

    if not all_chunks:
        logger.error("Нет чанков!")
        return

    # 3. Очищаем старый индекс
    working_dir = Path(SAVE_DIR) / f"{LLM_NAME}_{EMBEDDING_MODEL.replace('/', '_')}"
    for subdir in ["chunk_embeddings", "entity_embeddings", "fact_embeddings"]:
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

    # 4. Инициализируем HippoRAG и сохраняем чанки
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

    hipporag = HippoRAG(global_config=config)

    chunk_texts = [c.text for c in all_chunks]
    chunk_metas = [{
        "source_file": c.source_file,
        "doc_title": c.doc_title,
        "page_num": c.page_num,
        "section": c.section or "",
        "content": c.text,
    } for c in all_chunks]

    logger.info(f"Сохранение {len(chunk_texts)} чанков...")
    hipporag.chunk_embedding_store.insert_strings(chunk_texts, chunk_metas)
    logger.info(f"Чанки сохранены: {len(hipporag.chunk_embedding_store.embeddings)}")

    # 5. Получаем чанки
    chunk_to_rows = hipporag.chunk_embedding_store.get_all_id_to_rows()
    chunk_ids = list(chunk_to_rows.keys())

    # 6. Русский NER
    from hipporag.information_extraction.ner_russian import RussianNER
    from hipporag.utils.misc_utils import normalize_entity_name, is_valid_entity, \
        extract_entity_nodes, text_processing

    ner = RussianNER()
    chunk_entities = {}
    logger.info("Извлечение сущностей русским NER...")
    for chunk_id in chunk_ids:
        row = chunk_to_rows[chunk_id]
        text = row["content"] if isinstance(row, dict) else row
        ents = ner.extract(text)
        ents = [normalize_entity_name(e) for e in ents if is_valid_entity(normalize_entity_name(e))]
        chunk_entities[chunk_id] = list(dict.fromkeys(ents))

    total = sum(len(v) for v in chunk_entities.values())
    unique = set()
    for v in chunk_entities.values():
        unique.update(v)
    logger.info(f"Извлечено: {total} всего, {len(unique)} уникальных")

    # 7. Минимальные триплеты
    chunk_triples = []
    for chunk_id in chunk_ids:
        ents = chunk_entities.get(chunk_id, [])
        triples = []
        for e in ents:
            triples.append([e, "тема", "концепт"])
        chunk_triples.append(triples)

    processed = [[text_processing(t) for t in chunk_triples[i]] for i, chunk_id in enumerate(chunk_ids)]
    entity_nodes, chunk_triple_entities = extract_entity_nodes(processed)

    logger.info(f"Entity nodes: {len(entity_nodes)}")

    # 8. Entity embeddings
    logger.info("Создание entity embeddings...")
    hipporag.entity_embedding_store.insert_strings(entity_nodes)
    hipporag.fact_embedding_store.insert_strings(["placeholder"])

    # 9. Граф
    logger.info("Построение графа...")
    hipporag.node_to_node_stats = {}
    hipporag.ent_node_to_chunk_ids = {}
    hipporag.add_fact_edges(chunk_ids, processed)
    hipporag.add_passage_edges(chunk_ids, chunk_triple_entities)
    hipporag.add_synonymy_edges()
    hipporag.augment_graph()
    hipporag.save_igraph()
    logger.info(f"Граф: {hipporag.graph.vcount()} узлов, {hipporag.graph.ecount()} рёбер")

    # 10. Сохраняем OpenIE
    openie_data = {"docs": []}
    for ci, chunk_id in enumerate(chunk_ids):
        row = chunk_to_rows[chunk_id]
        passage = row["content"] if isinstance(row, dict) else row
        openie_data["docs"].append({
            "idx": chunk_id,
            "passage": passage,
            "extracted_entities": chunk_entities.get(chunk_id, []),
            "extracted_triples": processed[ci],
        })
    with open(str(openie_path), "w", encoding="utf-8") as f:
        json.dump(openie_data, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("СЕМАНТИЧЕСКОЕ ПЕРЕРАЗБИЕНИЕ ЗАВЕРШЕНО!")
    logger.info("=" * 60)
    logger.info(f"Чанков: {len(chunk_ids)}")
    logger.info(f"Entity nodes: {len(entity_nodes)}")
    logger.info(f"Граф: {hipporag.graph.vcount()} узлов, {hipporag.graph.ecount()} рёбер")


if __name__ == "__main__":
    main()
