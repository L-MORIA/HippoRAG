"""Совместимость guided-декодирования с разными поколениями `outlines`.

`outlines` — единственная внешняя зависимость, нужная
`hipporag.llm.transformers_offline` для ограничения вывода JSON-схемой.
Пакет менял API дважды, и декларация зависимости перестала соответствовать
действительности (аудит 04.11.2026, раунд 8):

* ``0.0.x`` — единственная ветка без Rust-расширения ``outlines_core``, но она
  требует ``numpy<2.0.0``: установка откатывает numpy и ломает ``pyarrow``/
  ``transformers`` в остальном окружении проекта;
* ``0.1.x``/``0.2.x`` — ``numpy`` больше не пинится, но нужен Rust-пакет
  ``outlines_core`` (колёса есть не для всех Python) и legacy-API
  ``outlines.generate.json``; ``0.2.x`` ограничен Python < 3.13;
* ``1.x`` — современный API ``outlines.Generator``; legacy-API удалён, пакет
  официально не выпускается для Python 3.14+.

Модуль скрывает различие API за одним интерфейсом, а отсутствие пакета
превращает в внятную ошибку вместо ``ImportError`` на уровне модуля
(из-за него модуль бэкенда нельзя было импортировать вовсе).
"""

import json
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Type

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Требование к `outlines`, соответствующее работоспособной конфигурации.
OUTLINES_REQUIREMENT = "outlines>=1.0,<2"

#: Подсказка по установке — показывается вместо голого ImportError.
OUTLINES_INSTALL_HINT = (
    "Guided-декодирование в `hipporag.llm.transformers_offline` требует пакет "
    f"`outlines` ({OUTLINES_REQUIREMENT}; официально поддерживается Python < 3.14). "
    "Установите его: pip install 'hipporag[offline]' "
    f"или pip install '{OUTLINES_REQUIREMENT}'."
)


class OutlinesUnavailableError(RuntimeError):
    """`outlines` не установлен либо установленная версия несовместима."""


def build_guided_generator(
    model, tokenizer, schema_model: Type[Any]
) -> Callable[[List[str], int], List[Any]]:
    """Строит генератор, ограничивающий вывод JSON-схемой `schema_model`.

    Поддерживаются оба API `outlines`:

    * ``>= 1.0`` — ``outlines.Generator`` (современный API);
    * ``< 1.0`` — ``outlines.generate.json`` (legacy-API, удалён в 1.0).

    Возвращается callable ``(prompts, max_tokens) -> list``: список объектов
    `schema_model`. Контракт одинаков для обеих веток, поэтому вызывающая
    сторона не различает версии `outlines`.

    Raises:
        OutlinesUnavailableError: пакет `outlines` не установлен или его
            импорт падает (например, из-за конфликта версий numpy).
    """
    outlines = _import_outlines()

    if _modern_api_available(outlines):
        return _build_modern_generator(model, tokenizer, schema_model)
    if _legacy_api_available(outlines):
        return _build_legacy_generator(model, tokenizer, schema_model)

    raise OutlinesUnavailableError(OUTLINES_INSTALL_HINT)


def _import_outlines():
    """Импортирует `outlines`, превращая любую ошибку импорта в понятную."""
    try:
        import outlines
    except Exception as exc:  # noqa: BLE001 — падать может любая причина, включая numpy-конфликт
        raise OutlinesUnavailableError(
            f"{OUTLINES_INSTALL_HINT} Исходная ошибка импорта: {exc}"
        ) from exc
    return outlines


def _modern_api_available(outlines) -> bool:
    """Есть ли современный API (`outlines.Generator`, `outlines>=1.0`)."""
    try:
        from outlines import Generator  # noqa: F401
        from outlines.models import Transformers  # noqa: F401
    except Exception:  # noqa: BLE001 — у 0.x этого API просто нет
        return False
    return True


def _legacy_api_available(outlines) -> bool:
    """Есть ли legacy-API (`outlines.generate.json`, `outlines<1.0`)."""
    try:
        import outlines.generate as generate
        import outlines.models as models
    except Exception:  # noqa: BLE001 — у 1.x этого модуля просто нет
        return False
    return hasattr(generate, "json") and hasattr(models, "Transformers")


def _build_modern_generator(model, tokenizer, schema_model: Type[Any]):
    """Ветка `outlines>=1.0`: ``outlines.Generator``."""
    from outlines import Generator
    from outlines.models import Transformers as OutlinesTransformers

    original_padding = getattr(tokenizer, "padding_side", "right")
    outlines_model = OutlinesTransformers(model, tokenizer)
    # `outlines` переводит токенизатор на left-padding «навсегда» (см.
    # `outlines/models/transformers.py`: `tokenizer.padding_side = "left"`).
    # Бэкенд не должен молча менять настройки чужого токенизатора, поэтому
    # возвращаем исходное значение: на время генерации padding_side всё равно
    # выставляется явно (см. `_left_padding`).
    tokenizer.padding_side = original_padding
    generator = Generator(outlines_model, schema_model)

    def run(prompts: List[str], max_tokens: int) -> List[Any]:
        with _left_padding(tokenizer):
            # `outlines` 1.x передаёт kwargs в `transformers.generate` как есть,
            # поэтому лимит токенов называется `max_new_tokens`.
            results = generator.batch(prompts, max_new_tokens=max_tokens)
        return [_as_model(result, schema_model) for result in results]

    return run


def _build_legacy_generator(model, tokenizer, schema_model: Type[Any]):
    """Ветка `outlines<1.0`: ``outlines.generate.json``."""
    import outlines.generate as generate
    import outlines.models as models

    outlines_model = models.Transformers(model, tokenizer)
    generator = _legacy_json_generator(generate, outlines_model, schema_model)

    def run(prompts: List[str], max_tokens: int) -> List[Any]:
        with _left_padding(tokenizer):
            # legacy-API `outlines` принимает лимит токенов как `max_tokens`.
            results = generator(prompts, max_tokens=max_tokens)
        return [_as_model(result, schema_model) for result in results]

    return run


def _legacy_json_generator(generate, outlines_model, schema_model: Type[Any]):
    """Строит legacy-генератор, ограниченный JSON-схемой.

    Предпочтителен `generate.regex` с регексом, собранным из схемы: он
    возвращает сырой текст, а разбор делает наш код — одинаково для обеих
    веток. У `generate.json` разбор встроен (`format_sequence = parse_raw`),
    и оборванная по `max_tokens` генерация роняет **весь батч**
    `pydantic.ValidationError` вместо того, чтобы отдать один негодный ответ.
    """
    try:
        from outlines_core.fsm.json_schema import build_regex_from_schema

        schema = json.dumps(schema_model.model_json_schema())
        return generate.regex(outlines_model, build_regex_from_schema(schema))
    except Exception:  # noqa: BLE001 — при неудаче откатываемся на штатный API outlines
        logger.warning(
            "Не удалось собрать регекс из JSON-схемы, используется `outlines.generate.json`: "
            "оборванная генерация будет ронять батч целиком."
        )
        return generate.json(outlines_model, schema_model)


def _as_model(result: Any, schema_model: Type[Any]) -> Any:
    """Приводит результат генерации к объекту `schema_model`.

    Современный API отдаёт JSON-строку, legacy — готовый pydantic-объект.
    Некорректный JSON не роняет весь батч: в лог уходит предупреждение, а
    потребитель (`openie_transformers_offline`) умеет переживать непарсящийся
    ответ — ровно как он переживал его до появления guided-декодирования.
    """
    if isinstance(result, schema_model):
        return result
    if isinstance(result, str):
        try:
            return schema_model.model_validate_json(result)
        except Exception as exc:  # noqa: BLE001 — причин много, все сводятся к «текст не по схеме»
            logger.warning(
                "Guided-декодирование вернуло текст, не соответствующий схеме "
                f"`{schema_model.__name__}`: {exc}. Ответ передаётся как есть."
            )
            return result
    return result


@contextmanager
def _left_padding(tokenizer) -> Iterator[None]:
    """Временно переключает `padding_side` токенизатора на `left`.

    `outlines` токенизирует батч с `padding=True`; при right-padding модель
    продолжает текст с pad-токена — тот же дефект, что и в free-form пути
    (аудит 04.11.2026, N-2).
    """
    original = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        yield
    finally:
        tokenizer.padding_side = original
