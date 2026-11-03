"""Контракт версий «индекс ↔ код ↔ кэш ↔ package-data» (H0-8).

Единое место, где объявлены версии форматов. До H0-8 версии жили в разных
модулях и сравнивались не везде, поэтому несовместимые данные читались молча:
индекс, собранный другой версией кода, отдавал результаты как ни в чём не
бывало, а записи QA-кэша до фиксов отдавались как «кэшированный ответ» (P1-9).

Что менять при изменении формата:

* ``INDEX_FORMAT_VERSION`` — раскладка каталога индекса, сериализация
  ``graph.pickle``, формат ``openie_results_*.json``, схема эмбеддингов.
* ``QA_CACHE_VERSION`` — поля записи QA-кэша и логика ответа (промпты).
* ``PACKAGE_DATA_VERSION`` — формат bundled-данных в ``astro_app/data``
  (``synonyms.json`` и манифест к нему).

Версия индекса пишется в ``index_manifest.json`` рядом с ``graph.pickle``
и проверяется при загрузке индекса. Версия package-data лежит в
``astro_app/data/data_manifest.json`` вместе с отпечатками файлов и
проверяется при каждом чтении данных. Несовпадение версии QA-кэша отсекает
устаревшие записи.

Модуль намеренно не импортирует ничего тяжёлого: его читают и ``astro_app``,
и ядро ``hipporag`` (лениво, внутри методов), и тесты.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

# ── Версии форматов ──────────────────────────────────────────────────────────

#: Версия формата каталога индекса. 1 — текущая раскладка: ``graph.pickle``
#: плюс ``chunk_embeddings``/``entity_embeddings``/``fact_embeddings``.
INDEX_FORMAT_VERSION = 1

#: Версия формата записи и логики ответа QA-кэша (P1-9). Увеличивается при
#: изменении полей записи или промптов: старые записи перестают читаться.
QA_CACHE_VERSION = 2

#: Версия формата bundled-данных в ``astro_app/data``.
PACKAGE_DATA_VERSION = 1

# ── Имена файлов контракта ───────────────────────────────────────────────────

#: Манифест формата индекса; лежит рядом с ``graph.pickle``.
INDEX_MANIFEST_NAME = "index_manifest.json"

#: Имя файла графа — по его наличию определяется, что индекс вообще есть.
INDEX_GRAPH_FILE = "graph.pickle"

#: Манифест package-data; лежит в каталоге данных пакета.
PACKAGE_DATA_MANIFEST_NAME = "data_manifest.json"

#: Каталог bundled-данных (работает и из исходников, и из установленного wheel).
PACKAGE_DATA_DIR = Path(str(resources.files("astro_app.data")))

#: Полный путь к манифесту package-data.
PACKAGE_DATA_MANIFEST = PACKAGE_DATA_DIR / PACKAGE_DATA_MANIFEST_NAME

# ── Состояния проверки индекса ───────────────────────────────────────────────

#: Индекса на диске нет — это нормальное состояние «ещё не индексировали».
INDEX_MISSING = "missing"
#: Индекс есть, но собран кодом до H0-8: формат не подтверждён.
INDEX_NO_MANIFEST = "no-manifest"
#: Индекс собран другим форматом — возможна тихая деградация.
INDEX_STALE = "stale"
#: Манифест есть, но не читается (битый JSON).
INDEX_CORRUPT = "corrupt"
#: Формат совпадает.
INDEX_OK = "ok"

#: Команда, которой можно «усыновить» индекс, собранный до H0-8.
ADOPT_COMMAND = "python scripts/index_manifest.py --adopt"


class PackageDataError(RuntimeError):
    """Package-data не соответствует версии или содержимому, ожидаемому кодом."""


# ── Общие помощники ──────────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Записать JSON атомарно: сначала во временный файл, потом ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _canonical_digest(data: object) -> str:
    """SHA-256 канонического JSON.

    Считается по разобранным данным, а не по байтам файла: так отпечаток не
    зависит от форматирования и переносов строк (CRLF на Windows против LF
    в CI) и ловит именно изменение содержимого.
    """
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Индекс ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IndexStatus:
    """Результат сверки формата индекса на диске с ожидаемым."""

    state: str
    working_dir: str
    expected_version: int
    found_version: int | None = None
    message: str = ""
    manifest: dict | None = None

    @property
    def is_problem(self) -> bool:
        """Требует ли состояние внимания пользователя."""
        return self.state in (INDEX_NO_MANIFEST, INDEX_STALE, INDEX_CORRUPT)


def index_manifest_path(working_dir: str | os.PathLike[str]) -> Path:
    """Путь к манифесту индекса в каталоге ``working_dir``."""
    return Path(working_dir) / INDEX_MANIFEST_NAME


def write_index_manifest(
    working_dir: str | os.PathLike[str],
    *,
    llm_name: str | None = None,
    embedding_model: str | None = None,
    counts: dict | None = None,
) -> Path:
    """Записать манифест формата индекса. Возвращает путь к файлу."""
    payload: dict = {
        "index_format_version": INDEX_FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if llm_name is not None:
        payload["llm_name"] = llm_name
    if embedding_model is not None:
        payload["embedding_model"] = embedding_model
    if counts:
        payload["counts"] = counts

    path = index_manifest_path(working_dir)
    _atomic_write_json(path, payload)
    return path


def read_index_manifest(working_dir: str | os.PathLike[str]) -> dict | None:
    """Прочитать манифест индекса. ``None`` — файла нет или он не читается."""
    path = index_manifest_path(working_dir)
    try:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check_index(
    working_dir: str | os.PathLike[str],
    graph_path: str | os.PathLike[str] | None = None,
) -> IndexStatus:
    """Сверить формат индекса на диске с :data:`INDEX_FORMAT_VERSION`.

    Индекса нет — состояние ``missing`` (не проблема). Индекс есть, но
    манифеста нет — ``no-manifest``: он собран кодом до H0-8, формат не
    подтверждён. Версия не совпала — ``stale``. Манифест битый — ``corrupt``.
    """
    graph = Path(graph_path) if graph_path is not None else Path(working_dir) / INDEX_GRAPH_FILE
    location = str(Path(working_dir))

    if not graph.exists():
        return IndexStatus(
            state=INDEX_MISSING,
            working_dir=location,
            expected_version=INDEX_FORMAT_VERSION,
            message="",
        )

    manifest = read_index_manifest(working_dir)
    if manifest is None:
        manifest_file = index_manifest_path(working_dir)
        reason = "манифест не читается" if manifest_file.exists() else "манифеста нет"
        message = (
            f"Формат индекса не подтверждён ({reason}): {manifest_file}\n"
            f"  Индекс собран кодом до H0-8 либо манифест повреждён. Код ожидает\n"
            f"  формат {INDEX_FORMAT_VERSION}. Если индекс заведомо совместим, отметьте его:\n"
            f"    {ADOPT_COMMAND}\n"
            f"  Иначе переиндексируйте (docs/HOW_TO_RUN.md). Работа продолжается."
        )
        state = INDEX_CORRUPT if manifest_file.exists() else INDEX_NO_MANIFEST
        return IndexStatus(
            state=state,
            working_dir=location,
            expected_version=INDEX_FORMAT_VERSION,
            message=message,
        )

    found = manifest.get("index_format_version")
    if found != INDEX_FORMAT_VERSION:
        message = (
            f"Индекс собран другим форматом: {found!r}, код ожидает {INDEX_FORMAT_VERSION}.\n"
            f"  Каталог: {location}\n"
            f"  Возможна тихая деградация результатов. Переиндексируйте (docs/HOW_TO_RUN.md)\n"
            f"  либо откатитесь на версию кода, которой соответствует индекс. Работа продолжается."
        )
        return IndexStatus(
            state=INDEX_STALE,
            working_dir=location,
            expected_version=INDEX_FORMAT_VERSION,
            found_version=found if isinstance(found, int) else None,
            message=message,
            manifest=manifest,
        )

    return IndexStatus(
        state=INDEX_OK,
        working_dir=location,
        expected_version=INDEX_FORMAT_VERSION,
        found_version=found,
        manifest=manifest,
    )


# ── Package-data ─────────────────────────────────────────────────────────────


def build_package_data_manifest(data_dir: str | os.PathLike[str] | None = None) -> dict:
    """Собрать манифест package-data по фактическим файлам каталога."""
    directory = Path(data_dir) if data_dir is not None else PACKAGE_DATA_DIR
    files: dict = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == PACKAGE_DATA_MANIFEST_NAME:
            continue
        with open(path, encoding="utf-8") as stream:
            content = json.load(stream)
        entry: dict = {"sha256": _canonical_digest(content)}
        if isinstance(content, (dict, list)):
            entry["entries"] = len(content)
        files[path.name] = entry
    return {"package_data_version": PACKAGE_DATA_VERSION, "files": files}


def write_package_data_manifest() -> Path:
    """Пересобрать и записать манифест package-data. Возвращает путь к файлу."""
    path = PACKAGE_DATA_MANIFEST
    _atomic_write_json(path, build_package_data_manifest())
    return path


def check_package_data() -> dict:
    """Проверить, что package-data соответствует коду.

    Возвращает манифест при успехе; бросает :class:`PackageDataError` с
    понятным объяснением при несовпадении версии, отсутствии файла или
    расхождении содержимого с отпечатком.
    """
    manifest_path = PACKAGE_DATA_MANIFEST
    if not manifest_path.exists():
        raise PackageDataError(
            f"Не найден манифест package-data: {manifest_path}\n"
            "  Соберите его: python scripts/update_package_data_manifest.py"
        )

    try:
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, ValueError) as error:
        raise PackageDataError(
            f"Манифест package-data не читается: {manifest_path}: {error}"
        ) from error

    if not isinstance(manifest, dict):
        raise PackageDataError(f"Манифест package-data должен быть объектом: {manifest_path}")

    found = manifest.get("package_data_version")
    if found != PACKAGE_DATA_VERSION:
        raise PackageDataError(
            f"Package-data несовместимы с кодом: манифест объявляет версию {found!r}, "
            f"код ожидает {PACKAGE_DATA_VERSION}.\n"
            f"  Манифест: {manifest_path}\n"
            "  Обновите данные и манифест: python scripts/update_package_data_manifest.py"
        )

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise PackageDataError(f"В манифесте package-data нет списка файлов: {manifest_path}")

    for name, meta in files.items():
        path = manifest_path.parent / name
        if not path.exists():
            raise PackageDataError(
                f"Package-data неполны: файл {name!r} отсутствует в {manifest_path.parent}\n"
                "  Соберите пакет заново: python scripts/update_package_data_manifest.py"
            )
        try:
            with open(path, encoding="utf-8") as stream:
                content = json.load(stream)
        except (OSError, ValueError) as error:
            raise PackageDataError(f"Файл package-data не читается: {path}: {error}") from error

        expected = (meta or {}).get("sha256")
        actual = _canonical_digest(content)
        if actual != expected:
            raise PackageDataError(
                f"Содержимое package-data не совпадает с манифестом: {name}\n"
                f"  ожидался sha256 {expected}, получен {actual}\n"
                "  Если данные изменены намеренно: python scripts/update_package_data_manifest.py"
            )

    return manifest
