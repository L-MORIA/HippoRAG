#!/usr/bin/env python3
"""Статус формата индекса и «усыновление» индексов, собранных до H0-8.

Индекс описывает себя манифестом ``index_manifest.json`` рядом с
``graph.pickle``. HippoRAG проверяет его при загрузке: если манифеста нет,
формат не подтверждён; если версия другая — предупреждение о возможной тихой
деградации. Этот скрипт показывает статус и позволяет осознанно отметить
существующий индекс как соответствующий текущему формату.

Запуск из корня репозитория:

    python scripts/index_manifest.py            # показать статус
    python scripts/index_manifest.py --adopt    # записать манифест для индекса

``--adopt`` ничего не перестраивает и не проверяет содержимое графа: это
ручное подтверждение «индекс собран совместимым кодом». При сомнении
переиндексируйте (docs/HOW_TO_RUN.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astro_app import config, versions  # noqa: E402 — путь к src добавляется выше


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Статус формата индекса HippoRAG")
    parser.add_argument(
        "--adopt",
        action="store_true",
        help="записать манифест текущей версии для существующего индекса",
    )
    parser.add_argument(
        "--working-dir",
        help="каталог индекса (по умолчанию собирается из config: SAVE_DIR/LLM_NAME/EMBEDDING_MODEL)",
    )
    args = parser.parse_args(argv)

    if args.working_dir:
        working_dir = Path(args.working_dir)
    else:
        from hipporag.utils.config_utils import build_working_dir

        working_dir = Path(
            build_working_dir(config.SAVE_DIR, config.LLM_NAME, config.EMBEDDING_MODEL)
        )

    status = versions.check_index(working_dir)
    print(f"Каталог индекса: {working_dir}")
    print(f"Состояние: {status.state} (ожидается формат {status.expected_version})")
    if status.message:
        print()
        print(status.message)

    if not args.adopt:
        return 0

    if status.state == versions.INDEX_MISSING:
        print("\nОТКАЗАНО: индекса нет — отмечать нечего.")
        return 1
    if status.state in (versions.INDEX_STALE, versions.INDEX_CORRUPT):
        print(
            "\nОТКАЗАНО: манифест уже есть и говорит о другом формате "
            f"({status.found_version!r}). Перезапись скрыла бы несовместимость.\n"
            "Если индекс действительно совместим, удалите манифест вручную и повторите."
        )
        return 1

    path = versions.write_index_manifest(
        working_dir,
        llm_name=config.LLM_NAME,
        embedding_model=config.EMBEDDING_MODEL,
    )
    print(f"\nМанифест записан: {path}")
    print(f"Формат индекса: {versions.INDEX_FORMAT_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
