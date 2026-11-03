"""Меню режимов поиска (H1-2).

Разделено с `astro_app.cli`, чтобы текст меню можно было проверять тестом без
поднятия индекса и без тяжёлых импортов (litellm инициализируется в `config`).

Контракт меню:

* две секции — «Рекомендуемые» и «Экспертные»;
* у режима по умолчанию (`modes.DEFAULT_MODE`) пометка «(по умолчанию)»;
* у каждого режима строка «когда: …» — когда этот режим уместен;
* звёздочка ⭐ только у рекомендованных режимов.
"""

from . import performance as perf
from .modes import DEFAULT_MODE, MODES, expert_modes, recommended_modes

#: Пометка режима, который по замерам медленнее 30 с (H1-3).
EXPERIMENTAL_MARK = " ⏱ экспериментальный"


def format_modes_menu() -> str:
    """Текст меню режимов (без печати — удобно для тестов)."""
    experimental = perf.experimental_modes(perf.load_baseline())

    def _mark(num: str) -> str:
        return EXPERIMENTAL_MARK if num in experimental else ""

    lines = ["Режимы поиска:", "  Рекомендуемые:"]
    for num, spec in recommended_modes().items():
        default_mark = " (по умолчанию)" if num == DEFAULT_MODE else ""
        lines.append(
            f"    {num} — {spec.menu}{' ⭐' if spec.star else ''}{default_mark}{_mark(num)}"
        )
        if spec.when:
            lines.append(f"         когда: {spec.when}")
        if spec.special:
            lines.append(f"         {spec.special}")

    lines.append("  Экспертные:")
    for num, spec in expert_modes().items():
        lines.append(f"    {num} — {spec.menu}{' ⭐' if spec.star else ''}{_mark(num)}")
        if spec.when:
            lines.append(f"         когда: {spec.when}")
        if spec.special:
            lines.append(f"         {spec.special}")

    lines.append("  t — Тестирование всех методов по кэшу")
    lines.append("  e — Экспорт кэша в JSONL (для обучения)")
    lines.append("  q — Выход")
    return "\n".join(lines)


def print_modes_menu() -> None:
    """Печать меню режимов."""
    print(format_modes_menu())


def default_mode_label() -> str:
    """Подпись режима по умолчанию для приглашения ввода."""
    return MODES[DEFAULT_MODE].label
