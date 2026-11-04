"""Константы и конфигурация приложения.

Содержимое дословно повторяет константы main.py (шаг 1 миграции MAIN_SPLIT_PLAN.md);
main.py переключается на этот модуль на шаге 7. Импорт модуля выполняет ту же
настройку окружения, что и импорт main.py (порядок: load_dotenv → чтение env).
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

SAVE_DIR = "outputs/astrology_test"
LLM_NAME = os.getenv("LLM_NAME", "gemma-4-e4b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

STOP_WORDS = {
    "что",
    "как",
    "все",
    "она",
    "так",
    "его",
    "только",
    "мне",
    "было",
    "меня",
    "нет",
    "для",
    "квартир",
    "дом",
    "комнат",
    "себе",
    "этот",
    "можно",
    "если",
    "чтобы",
    "когда",
    "даже",
    "уже",
    "есть",
    "при",
    "том",
    "или",
    "из",
}

SENTENCE_START = re.compile(r'^[А-ЯA-Z«"\']')


def load_hipporag(
    llm_name: str | None = None,
    *,
    save_dir: str | None = None,
    llm_base_url: str | None = None,
    embedding_model_name: str | None = None,
    qa_top_k: int | None = None,
    qa_strict_grounding: bool | None = None,
):
    """Создаёт HippoRAG с конфигурацией main() (без интерактивного выбора модели).

    Выбор модели из MODEL_OPTIONS остаётся в cli.py; сюда передаётся уже
    выбранное имя (или None — берётся LLM_NAME из окружения).

    Необязательные параметры нужны eval-скриптам: они позволяют прогнать
    golden set на конкретном индексе и сравнить `qa_top_k`, не меняя `.env`.
    """
    from hipporag import HippoRAG  # лениво: импорт пакета тянет тяжёлые зависимости
    from hipporag.utils.config_utils import BaseConfig

    overrides = {}
    if qa_top_k is not None:
        overrides["qa_top_k"] = qa_top_k
    if qa_strict_grounding is not None:
        overrides["qa_strict_grounding"] = qa_strict_grounding

    config = BaseConfig(
        save_dir=save_dir or SAVE_DIR,
        llm_name=llm_name or LLM_NAME,
        llm_base_url=llm_base_url or LLM_BASE_URL,
        embedding_model_name=embedding_model_name or EMBEDDING_MODEL,
        linking_top_k=100,
        retrieval_top_k=200,
        passage_node_weight=0.05,
        embedding_batch_size=8,
        **overrides,
    )
    return HippoRAG(global_config=config)
