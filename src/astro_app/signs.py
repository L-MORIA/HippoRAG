"""Знаковая инфраструктура: доминирующий знак чанка, счёт знаков, keyword-буст.

Перенос из main.py (шаг 6) — общие хелперы, используемые 9 модулями режимов.
Зависят от констант знаков из ..query (байт-в-байт перенос 1322-1412).
"""

from .query import _ALL_SIGN_FORMS_RE, FORM_TO_SIGN, ZODIAC_FORMS, ZODIAC_SIGNS

_SIGN_DOMINANCE = None  # chunk_idx → dominant_zodiac_sign


def _build_sign_index(chunk_store):
    """Предвычисляем, какой знак зодиака доминирует в каждом чанке."""
    global _SIGN_DOMINANCE
    if _SIGN_DOMINANCE is not None:
        return _SIGN_DOMINANCE
    _SIGN_DOMINANCE = {}
    for idx, text in enumerate(chunk_store.texts):
        tl = text.lower()
        counts = {}
        for sign in ZODIAC_SIGNS:
            c = tl.count(sign)
            if c > 0:
                counts[sign] = c
        if counts:
            dominant = max(counts, key=counts.get)
            _SIGN_DOMINANCE[idx] = dominant
    return _SIGN_DOMINANCE


def _sign_in_query(q_lower):
    """Найти знак зодиака в запросе (любой падеж) → именительный падеж."""
    match = _ALL_SIGN_FORMS_RE.search(q_lower.lower())
    if match:
        return FORM_TO_SIGN[match.group(1)]
    return None


def _count_sign(tl, sign):
    """Сколько раз знак зодиака упомянут в тексте (все падежи)."""
    total = 0
    for form in ZODIAC_FORMS.get(sign, [sign]):
        total += tl.count(form)
    return total


def _keyword_boost(tl, q_lower, terms):
    boost = 0.0
    if q_lower in tl:
        boost += 1.0
    matches = sum(1 for t in terms if t in tl)
    if terms:
        boost += 0.3 * min(matches, 5) / max(len(terms), 1)
    if matches >= max(2, len(terms) - 1):
        boost += 2.0
    for t in terms:
        if len(t) <= 2:
            continue
        if tl.count(t) >= 3:
            boost += 1.5
        if t in tl[:200]:
            boost += 0.5
    # Co-occurrence boost: если все термы запроса встречаются в радиусе 300 символов
    if len(terms) >= 2:
        for i, t1 in enumerate(terms):
            for t2 in terms[i + 1 :]:
                if t1 in tl and t2 in tl:
                    pos1 = tl.find(t1)
                    pos2 = tl.find(t2)
                    if abs(pos1 - pos2) < 300:
                        boost += 2.0  # сильный буст когда термы рядом
    # Определяем знак зодиака в запросе (любой падеж)
    sign_in_query = _sign_in_query(q_lower)
    if sign_in_query and _count_sign(tl, sign_in_query) >= 2:
        boost += 1.5
    return boost


def _chunk_sign_score(chunk_store, idx, q_lower):
    """Возвращает скор доминирования знака зодиака для чанка."""
    sign_in_query = _sign_in_query(q_lower)
    if not sign_in_query:
        return 0.0
    tl = chunk_store.texts[idx].lower()
    sign_count = _count_sign(tl, sign_in_query)
    if sign_count == 0:
        return -3.0  # штраф: знак вообще не упомянут
    # Считаем все знаки (все падежи)
    total = sum(_count_sign(tl, s) for s in ZODIAC_SIGNS)
    if total == 0:
        return 0.0
    ratio = sign_count / total
    if ratio >= 0.5:
        return 4.0  # знак доминирует (>50% всех упоминаний)
    elif ratio >= 0.2:
        return 2.0  # знак заметно присутствует
    return 0.5
