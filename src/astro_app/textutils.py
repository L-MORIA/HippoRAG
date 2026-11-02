"""Текстовые утилиты отображения и токенизации (перенос из main.py, шаг 2)."""

import re


def show_text_safe(text, max_len=600):
    """Обрезать текст на границе предложения, пометить обрыв."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    last_period = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("! "), cut.rfind("? "))
    if last_period > max_len * 0.3:
        return cut[: last_period + 1] + "\n[...]"
    return cut.rsplit(" ", 1)[0] + "\n[...]"


def fix_chunk_start(text):
    """
    Если чанк начинается с середины предложения (первая буква строчная),
    найти первый обрыв предложения в начале и отрезать его.
    """
    t = text.strip()
    if not t:
        return text, False

    # Первая буква строчная → обрыв
    if t[0].islower():
        # Попытаться найти начало первого полного предложения
        for sep in [". ", ".\n", "? ", "! ", "\n\n"]:
            idx = t.find(sep)
            if 20 < idx < 400:
                return t[idx + len(sep) :], True
        return text, True  # не нашли границу — просто помечаем

    return text, False


def _tokenize(text):
    return re.findall(r"[а-яёa-z0-9]+", text.lower())
