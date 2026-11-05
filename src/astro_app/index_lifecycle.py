"""H2-2: жизненный цикл индекса.

Модуль отвечает за операции, которые раньше делались только вручную:

* статус индекса и идущей операции (``status``);
* список документов, попавших в индекс (``documents``);
* резервная копия каталога индекса (``backup``);
* восстановление из копии с откатом при ошибке (``restore``);
* удаление документов (``delete_documents``);
* переиндексация PDF с предварительным бэкапом (``reindex``).

Модуль намеренно не импортирует FastAPI и не знает про HTTP: его можно
использовать и из API, и из CLI, и в тестах. Связь с ``HippoRAG`` передаётся
колбэками ``get_hipporag``/``close_hipporag``/``reload_hipporag``.

Границы безопасности:

* имя резервной копии — только basename с расширением ``.zip``; выход за
  каталог бэкапов запрещён;
* restore распаковывает копию во временный каталог, проверяет манифест и
  ``graph.pickle`` и только потом подменяет каталог индекса;
* при ошибке reload старый каталог возвращается на место;
* у чанка, который принадлежит нескольким документам, удаляется только
  metadata-запись удаляемого источника: сам чанк остаётся, иначе вместе с
  одним документом пропали бы данные другого.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

from . import versions

logger = logging.getLogger(__name__)

#: Имя каталога резервных копий рядом с каталогом индекса.
BACKUP_DIR_NAME = "backups"

#: Служебные файлы/каталоги, которые не попадают в резервную копию.
_SKIP_NAMES = {".DS_Store", "Thumbs.db", "__pycache__"}

_SAFE_LABEL = re.compile(r"[^A-Za-z0-9_.-]+")


def _now_iso() -> str:
    """Текущее время в UTC в формате ISO 8601."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _timestamp() -> str:
    """Компактная метка времени для имён файлов."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _metadata_entries(meta: Any) -> List[Dict[str, Any]]:
    """Привести metadata чанка к списку словарей.

    Поддерживаются текущий формат (список словарей), старый (один словарь)
    и JSON-строка: так же, как это делает ``EmbeddingStore``.
    """
    if meta is None or meta == "":
        return []
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (TypeError, ValueError):
            return []
    if isinstance(meta, dict):
        return [meta]
    if isinstance(meta, list):
        return [item for item in meta if isinstance(item, dict)]
    return []


def _source_name(value: Any) -> str:
    """Имя файла из metadata: полный путь тоже принимается."""
    text = str(value or "").strip()
    if not text:
        return ""
    # Windows-путь или POSIX-путь — в обоих случаях нужно последнее звено.
    return Path(text.replace("\\", "/")).name


def _matches_source(candidate: str, targets: Set[str]) -> bool:
    """Совпадает ли источник с одним из целевых имён."""
    if not candidate:
        return False
    return candidate in targets or _source_name(candidate) in targets


@dataclass
class OperationState:
    """Состояние последней/текущей операции с индексом."""

    state: str = "idle"  # idle | running | error
    operation: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "operation": self.operation,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "detail": dict(self.detail),
        }


class IndexLifecycle:
    """Операции над каталогом индекса HippoRAG.

    Parameters
    ----------
    index_dir:
        Каталог конкретного индекса (``HippoRAG.working_dir``), в котором
        лежат ``graph.pickle``, ``index_manifest.json`` и parquet-хранилища.
    backup_dir:
        Куда складывать резервные копии. По умолчанию — ``backups`` рядом
        с каталогом индекса.
    get_hipporag:
        Возвращает текущий объект ``HippoRAG`` или ``None``.
    close_hipporag:
        Закрывает ресурсы старого объекта перед подменой каталога.
    reload_hipporag:
        Создаёт новый объект ``HippoRAG`` после восстановления.
    """

    def __init__(
        self,
        index_dir: str | os.PathLike[str],
        backup_dir: str | os.PathLike[str] | None = None,
        *,
        get_hipporag: Optional[Callable[[], Any]] = None,
        close_hipporag: Optional[Callable[[], None]] = None,
        reload_hipporag: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.index_dir = Path(index_dir).resolve()
        self.backup_dir = (
            Path(backup_dir).resolve()
            if backup_dir is not None
            else self.index_dir.parent / BACKUP_DIR_NAME
        )
        self._get_hipporag = get_hipporag
        self._close_hipporag = close_hipporag
        self._reload_hipporag = reload_hipporag
        self._state = OperationState()
        self._lock = threading.RLock()
        self._busy = False

    # ── статус ──────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Снимок состояния индекса и последней операции."""
        manifest = versions.read_index_manifest(self.index_dir) or {}
        check = versions.check_index(self.index_dir)
        chunks = facts = documents = 0
        hipporag = self._hipporag()
        if hipporag is not None:
            chunks = self._count_store(hipporag, "chunk_embedding_store")
            facts = self._count_store(hipporag, "fact_embedding_store")
            try:
                documents = len(self.documents(hipporag=hipporag))
            except Exception as error:  # noqa: BLE001 — статус не должен падать
                logger.debug("Не удалось посчитать документы: %s", error)

        return {
            **self._state.as_dict(),
            "index": {
                "working_dir": str(self.index_dir),
                "format_version": manifest.get("index_format_version"),
                "expected_format_version": versions.INDEX_FORMAT_VERSION,
                "format_state": check.state,
                "created_at": manifest.get("created_at"),
                "llm_name": manifest.get("llm_name"),
                "embedding_model": manifest.get("embedding_model"),
                "counts": manifest.get("counts") or {},
                "chunks": chunks,
                "facts": facts,
                "documents": documents,
            },
            "backups": len(self.list_backups()),
        }

    def documents(self, *, hipporag: Any = None) -> List[Dict[str, Any]]:
        """Список документов в индексе, сгруппированный по ``source_file``."""
        hipporag = hipporag if hipporag is not None else self._hipporag()
        if hipporag is None:
            raise RuntimeError("HippoRAG не инициализирован")

        store = getattr(hipporag, "chunk_embedding_store", None)
        if store is None:
            raise RuntimeError("У HippoRAG нет chunk_embedding_store")

        rows = self._all_rows(store)
        aggregated: Dict[str, Dict[str, Any]] = {}
        for row in rows.values():
            for entry in _metadata_entries(row.get("metadata")):
                source = _source_name(entry.get("source_file"))
                if not source:
                    continue
                item = aggregated.setdefault(
                    source,
                    {
                        "source_file": source,
                        "doc_title": "",
                        "pages": set(),
                        "chunks": 0,
                    },
                )
                title = str(entry.get("doc_title") or "").strip()
                if title and not item["doc_title"]:
                    item["doc_title"] = title
                page = entry.get("page_num")
                if page is not None:
                    item["pages"].add(page)
                item["chunks"] += 1

        result = []
        for item in aggregated.values():
            result.append(
                {
                    "source_file": item["source_file"],
                    "doc_title": item["doc_title"],
                    "pages": len(item["pages"]),
                    "chunks": item["chunks"],
                }
            )
        return sorted(result, key=lambda item: item["source_file"].lower())

    # ── резервные копии ─────────────────────────────────────────────────────

    def backup(self, label: Optional[str] = None) -> Dict[str, Any]:
        """Создать zip-копию каталога индекса.

        Во время другой операции бэкап запрещён: иначе копия может поймать
        наполовину записанный parquet.
        """
        with self._lock:
            if self._busy:
                raise RuntimeError("Нельзя делать бэкап во время другой операции с индексом")
            return self._create_backup(label)

    def _create_backup(self, label: Optional[str] = None) -> Dict[str, Any]:
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Каталог индекса не найден: {self.index_dir}")
        if not (self.index_dir / versions.INDEX_GRAPH_FILE).exists():
            raise FileNotFoundError(
                f"В каталоге индекса нет {versions.INDEX_GRAPH_FILE}: {self.index_dir}"
            )

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        name = self._backup_name(label)
        archive = self.backup_dir / name
        created_at = _now_iso()
        files = 0
        total_bytes = 0

        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in self._walk_index_files():
                arcname = path.relative_to(self.index_dir).as_posix()
                bundle.write(path, arcname)
                files += 1
                total_bytes += path.stat().st_size

        problems = self._verify_archive(archive)
        if problems:
            archive.unlink(missing_ok=True)
            raise ValueError("Резервная копия повреждена: " + "; ".join(problems))

        record = {
            "name": name,
            "path": str(archive),
            "created_at": created_at,
            "files": files,
            "bytes": total_bytes,
            "label": label or "",
            "source_dir": str(self.index_dir),
        }
        self._write_backup_sidecar(record)
        logger.info("Создан бэкап индекса: %s (%d файлов, %d байт)", archive, files, total_bytes)
        return record

    def list_backups(self) -> List[Dict[str, Any]]:
        """Список резервных копий, новые сверху."""
        if not self.backup_dir.exists():
            return []
        records = []
        for archive in sorted(
            self.backup_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True
        ):
            record = self._read_backup_sidecar(archive)
            if record is None:
                record = {
                    "name": archive.name,
                    "path": str(archive),
                    "created_at": datetime.fromtimestamp(archive.stat().st_mtime, tz=timezone.utc)
                    .replace(microsecond=0)
                    .isoformat(),
                    "files": 0,
                    "bytes": archive.stat().st_size,
                    "label": "",
                    "source_dir": str(self.index_dir),
                }
            records.append(record)
        return records

    def restore(self, name: str) -> Dict[str, Any]:
        """Восстановить индекс из копии.

        Копия распаковывается во временный каталог, проверяется манифест,
        затем каталог индекса подменяется. Если reload падает, старый каталог
        возвращается на место.
        """
        archive = self._resolve_backup(name)
        with self._lock:
            self._begin("restore", {"backup": archive.name})
            try:
                result = self._restore_locked(archive)
                self._finish(detail=result)
                return result
            except Exception as error:
                self._fail(error)
                raise

    def _restore_locked(self, archive: Path) -> Dict[str, Any]:
        staging = Path(tempfile.mkdtemp(prefix=".restore-", dir=str(self.index_dir.parent)))
        old_dir = self.index_dir.parent / f"{self.index_dir.name}.old-{_timestamp()}"
        try:
            with zipfile.ZipFile(archive) as bundle:
                problems = bundle.testzip()
                if problems is not None:
                    raise ValueError(f"Повреждённый файл в копии: {problems}")
                bundle.extractall(staging)

            graph = staging / versions.INDEX_GRAPH_FILE
            if not graph.exists():
                raise ValueError(f"В копии нет {versions.INDEX_GRAPH_FILE}")
            check = versions.check_index(staging)
            if check.state != versions.INDEX_OK:
                raise ValueError(
                    "Копия не прошла проверку формата: " f"{check.state}. {check.message}".strip()
                )

            if self._close_hipporag is not None:
                self._close_hipporag()

            if self.index_dir.exists():
                if old_dir.exists():
                    shutil.rmtree(old_dir)
                self.index_dir.rename(old_dir)
            staging.rename(self.index_dir)

            try:
                if self._reload_hipporag is not None:
                    self._reload_hipporag()
            except Exception:
                # Возвращаем старый индекс и пытаемся поднять его снова.
                failed = self.index_dir.parent / f"{self.index_dir.name}.failed-{_timestamp()}"
                if self.index_dir.exists():
                    self.index_dir.rename(failed)
                if old_dir.exists():
                    old_dir.rename(self.index_dir)
                if self._reload_hipporag is not None:
                    try:
                        self._reload_hipporag()
                    except Exception as reload_error:  # noqa: BLE001
                        logger.error("Не удалось поднять старый индекс: %s", reload_error)
                shutil.rmtree(failed, ignore_errors=True)
                raise

            shutil.rmtree(old_dir, ignore_errors=True)
            return {
                "restored": archive.name,
                "working_dir": str(self.index_dir),
                "format_state": versions.check_index(self.index_dir).state,
            }
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # ── документы ───────────────────────────────────────────────────────────

    def delete_documents(self, source_files: Sequence[str]) -> Dict[str, Any]:
        """Удалить документы по именам ``source_file``.

        Чанк удаляется, только если все его источники входят в удаляемый
        набор. Общие чанки (один текст из двух PDF) остаются: иначе вместе
        с одним документом исчезли бы данные другого.
        """
        targets = {_source_name(name) for name in source_files if _source_name(name)}
        if not targets:
            raise ValueError("Не указан ни один документ")

        with self._lock:
            self._begin("delete", {"source_files": sorted(targets)})
            try:
                result = self._delete_locked(targets)
                self._finish(detail=result)
                return result
            except Exception as error:
                self._fail(error)
                raise

    def _delete_locked(self, targets: Set[str]) -> Dict[str, Any]:
        hipporag = self._require_hipporag()
        store = getattr(hipporag, "chunk_embedding_store", None)
        if store is None:
            raise RuntimeError("У HippoRAG нет chunk_embedding_store")

        texts, shared_ids = self._chunk_texts_for_sources(hipporag, targets)
        existing = {item["source_file"] for item in self.documents(hipporag=hipporag)}
        missing = sorted(targets - existing)

        shared_updated = 0
        skipped_shared = 0
        remove_metadata = getattr(store, "remove_metadata_entries", None)
        if callable(remove_metadata):
            for hash_id in shared_ids:
                if remove_metadata(hash_id, targets):
                    shared_updated += 1
        else:
            skipped_shared = len(shared_ids)

        if texts:
            hipporag.delete(texts)
            # delete() сам вызывает save_igraph(), но при отсутствии изменений
            # манифест мог не обновиться — записываем его явно.
            self._write_manifest(hipporag)
        elif shared_updated:
            # Чанки остались, но metadata изменилась — манифест и граф не
            # меняются, поэтому достаточно перезаписать parquet (это уже
            # сделал remove_metadata_entries).
            logger.info("Обновлены metadata %d общих чанков", shared_updated)

        return {
            "deleted_documents": sorted(targets & existing),
            "missing_documents": missing,
            "deleted_chunks": len(texts),
            "shared_chunks_updated": shared_updated,
            "skipped_shared_chunks": skipped_shared,
        }

    def reindex(self, pdf_paths: Sequence[str]) -> Dict[str, Any]:
        """Переиндексировать указанные PDF с бэкапом текущего индекса.

        Сначала создаётся копия, затем удаляются старые чанки этих файлов
        (кроме общих) и вызывается ``index_pdfs``. Так повторная загрузка
        того же PDF не оставляет дубликатов.
        """
        paths = [str(path) for path in pdf_paths if str(path).strip()]
        if not paths:
            raise ValueError("Не указан ни один PDF")
        missing = [path for path in paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError("Файлы не найдены: " + ", ".join(missing))

        with self._lock:
            self._begin("reindex", {"pdf_paths": paths})
            try:
                backup = self._create_backup("before-reindex")
                hipporag = self._require_hipporag()
                sources = {_source_name(path) for path in paths}
                before = self._count_store(hipporag, "chunk_embedding_store")
                deleted = self._delete_locked(sources)
                hipporag.index_pdfs(paths)
                after = self._count_store(hipporag, "chunk_embedding_store")
                self._write_manifest(hipporag)
                result = {
                    "backup": backup,
                    "deleted": deleted,
                    "pdf_paths": paths,
                    "chunks_before": before,
                    "chunks_after": after,
                }
                self._finish(detail=result)
                return result
            except Exception as error:
                self._fail(error)
                raise

    # ── внутренние помощники ────────────────────────────────────────────────

    def _hipporag(self) -> Any:
        if self._get_hipporag is None:
            return None
        try:
            return self._get_hipporag()
        except Exception as error:  # noqa: BLE001
            logger.debug("get_hipporag упал: %s", error)
            return None

    def _require_hipporag(self) -> Any:
        hipporag = self._hipporag()
        if hipporag is None:
            raise RuntimeError("HippoRAG не инициализирован")
        return hipporag

    @staticmethod
    def _count_store(hipporag: Any, attribute: str) -> int:
        store = getattr(hipporag, attribute, None)
        if store is None:
            return 0
        try:
            return len(store.get_all_ids())
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _all_rows(store: Any) -> Dict[str, Dict[str, Any]]:
        getter = getattr(store, "get_all_id_to_rows", None)
        if callable(getter):
            return dict(getter())
        rows: Dict[str, Dict[str, Any]] = {}
        for hash_id in store.get_all_ids():
            rows[hash_id] = store.get_row(hash_id)
        return rows

    def _chunk_texts_for_sources(
        self, hipporag: Any, targets: Set[str]
    ) -> tuple[List[str], List[str]]:
        """Вернуть (тексты чанков на удаление, id общих чанков).

        Общий чанк нельзя удалять целиком — у него есть metadata другого
        документа. Для него возвращается hash_id, чтобы вызывающий код убрал
        только запись об удаляемом источнике.
        """
        store = getattr(hipporag, "chunk_embedding_store", None)
        if store is None:
            raise RuntimeError("У HippoRAG нет chunk_embedding_store")
        texts: List[str] = []
        shared_ids: List[str] = []
        for hash_id, row in self._all_rows(store).items():
            entries = _metadata_entries(row.get("metadata"))
            sources = {
                _source_name(entry.get("source_file"))
                for entry in entries
                if _source_name(entry.get("source_file"))
            }
            if not (sources & targets):
                continue
            if sources <= targets:
                text = str(row.get("content") or "")
                if text:
                    texts.append(text)
            else:
                shared_ids.append(hash_id)
        return texts, shared_ids

    def _write_manifest(self, hipporag: Any) -> None:
        """Обновить счётчики в манифесте после операции."""
        writer = getattr(hipporag, "_write_index_manifest", None)
        if callable(writer):
            writer()
            return
        versions.write_index_manifest(self.index_dir)

    def _walk_index_files(self) -> Iterable[Path]:
        for root, dirs, files in os.walk(self.index_dir):
            root_path = Path(root)
            dirs[:] = [
                name
                for name in dirs
                if name not in _SKIP_NAMES
                and not name.startswith(".restore-")
                and not name.startswith(f"{self.index_dir.name}.old-")
                and (root_path / name).resolve() != self.backup_dir
            ]
            for name in files:
                if name in _SKIP_NAMES or name.endswith(".tmp"):
                    continue
                path = root_path / name
                if path.resolve().is_relative_to(self.backup_dir):
                    continue
                yield path

    def _backup_name(self, label: Optional[str]) -> str:
        base = f"index-{_timestamp()}"
        if label:
            cleaned = _SAFE_LABEL.sub("-", label).strip(".-")
            if cleaned:
                base = f"{base}-{cleaned}"
        candidate = f"{base}.zip"
        counter = 1
        while (self.backup_dir / candidate).exists():
            candidate = f"{base}-{counter}.zip"
            counter += 1
        return candidate

    def _resolve_backup(self, name: str) -> Path:
        candidate = Path(str(name))
        if candidate.name != str(name) or candidate.suffix.lower() != ".zip":
            raise ValueError("Имя резервной копии должно быть basename с расширением .zip")
        archive = (self.backup_dir / candidate).resolve()
        if not archive.is_relative_to(self.backup_dir):
            raise ValueError("Путь к резервной копии выходит за каталог бэкапов")
        if not archive.exists():
            raise FileNotFoundError(f"Резервная копия не найдена: {archive.name}")
        return archive

    def _verify_archive(self, archive: Path) -> List[str]:
        problems: List[str] = []
        try:
            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
                if bundle.testzip() is not None:
                    problems.append("повреждённый файл")
                if versions.INDEX_GRAPH_FILE not in names:
                    problems.append(f"нет {versions.INDEX_GRAPH_FILE}")
                if versions.INDEX_MANIFEST_NAME not in names:
                    problems.append(f"нет {versions.INDEX_MANIFEST_NAME}")
        except (OSError, zipfile.BadZipFile) as error:
            problems.append(str(error))
        return problems

    def _sidecar_path(self, archive: Path) -> Path:
        return archive.with_suffix(archive.suffix + ".json")

    def _write_backup_sidecar(self, record: Dict[str, Any]) -> None:
        path = self._sidecar_path(Path(record["path"]))
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _read_backup_sidecar(self, archive: Path) -> Optional[Dict[str, Any]]:
        path = self._sidecar_path(archive)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _begin(self, operation: str, detail: Dict[str, Any]) -> None:
        if self._busy:
            raise RuntimeError(f"Операция уже выполняется: {self._state.operation or 'unknown'}")
        self._busy = True
        self._state = OperationState(
            state="running",
            operation=operation,
            started_at=_now_iso(),
            detail=detail,
        )

    def _finish(self, detail: Optional[Dict[str, Any]] = None) -> None:
        self._state.state = "idle"
        self._state.finished_at = _now_iso()
        self._state.error = None
        if detail is not None:
            self._state.detail = detail
        self._busy = False

    def _fail(self, error: BaseException) -> None:
        self._state.state = "error"
        self._state.finished_at = _now_iso()
        self._state.error = str(error)
        self._busy = False
        logger.exception("Операция с индексом завершилась ошибкой: %s", error)
