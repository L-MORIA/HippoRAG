"""Режимы sign_zones — перенос из main.py (шаг 6, AST-извлечение из git HEAD)."""

import re

from .. import config
from ..config import LLM_BASE_URL
from ..primitives import _get_chunk_meta
from ..query import _SIGN_RE, ZODIAC_FORMS, ZODIAC_SIGNS, _word_re
from ..signs import (
    _ACCUSATIVE,
    _CLASS_NAMES,
    _SIGN_ZONE_STRONG,
    _SIGN_ZONE_WEAK,
    _ZONE_TEXT_STEMS,
    SIGN_PLANET,
    ZONE_ALIASES,
    _count_sign,
    _sign_in_query,
)

_SENTENCE_INDEX = None


_SENTENCE_INDEX_KEY = None


def _sentence_index(chunk_store):
    """Кэш предложений корпуса: [(lower, original, chunk_idx)]."""
    global _SENTENCE_INDEX, _SENTENCE_INDEX_KEY
    key = (id(chunk_store), len(chunk_store.texts))
    if _SENTENCE_INDEX is not None and _SENTENCE_INDEX_KEY == key:
        return _SENTENCE_INDEX
    sentences = []
    for idx, text in enumerate(chunk_store.texts):
        for sentence in re.split(r"(?<=[.!?])\s+", str(text)):
            sentence = sentence.strip()
            if len(sentence) > 15:
                sentences.append((sentence.lower(), sentence, idx))
    _SENTENCE_INDEX = sentences
    _SENTENCE_INDEX_KEY = key
    return sentences


def extract_sign_full(hipporag, query):
    """
    Извлечь ВЕСЬ материал по знаку зодиака, сохранить в DOCX/TXT.
    """
    from pathlib import Path

    from docx import Document
    from docx.shared import Pt

    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []

    n_chunks = len(chunk_store.texts)
    q_lower = query.lower()
    sign = _sign_in_query(q_lower)
    if not sign:
        return [("Не указан знак зодиака в запросе", 0, "", 0, 0, 0)]

    # Собираем ВСЕ чанки, где упоминается знак
    sign_chunks = []
    for idx in range(n_chunks):
        tl = chunk_store.texts[idx].lower()
        if _count_sign(tl, sign) > 0:
            source, page = _get_chunk_meta(chunk_store, idx)
            sign_chunks.append((idx, source, page, chunk_store.texts[idx]))

    if not sign_chunks:
        return [(f"Материал по знаку '{sign}' не найден", 0, "", 0, 0, 0)]

    # Сортируем по источнику и странице
    sign_chunks.sort(key=lambda x: (x[1], x[2], x[0]))

    # Склеиваем
    full_text_parts = []
    prev_source = None
    prev_page = 0
    for idx, source, page, text in sign_chunks:
        if prev_source and source != prev_source:
            full_text_parts.append(f"\n\n=== {source} ===\n")
        if page != prev_page + 1 and prev_source:
            full_text_parts.append(f"\n[...стр.{page}]\n")
        full_text_parts.append(text)
        prev_source = source
        prev_page = page

    full_text = "\n".join(full_text_parts)
    total_chars = len(full_text)
    total_chunks = len(sign_chunks)

    # Сохраняем в DOCX
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{sign}.docx"

    doc = Document()
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Arial"
    font.size = Pt(11)

    doc.add_heading(f"Материал по знаку: {sign.upper()}", level=1)
    doc.add_paragraph("Источник: книга Ирины Максимовой «Астрофэншуй квартиры»")
    doc.add_paragraph(f"Всего чанков: {total_chunks}, символов: {total_chars}")
    doc.add_paragraph("")

    prev_source = None
    for idx, source, page, text in sign_chunks:
        if prev_source and source != prev_source:
            doc.add_page_break()
            doc.add_heading(f"{source}", level=2)
        doc.add_paragraph(text)
        prev_source = source

    doc.save(str(fname))
    size_kb = total_chars // 1024

    result_text = (
        f"Файл сохранён: {fname}\n"
        f"Знак: {sign}\n"
        f"Чанков: {total_chunks} | Символов: {total_chars} (~{size_kb} KB)\n"
        f"--- Начало материала ({min(1500, total_chars)} симв.) ---\n\n"
        f"{full_text[:1500]}"
    )

    return [(result_text, total_chunks, sign, total_chars, 0, 0)]


def extract_sign_polished(hipporag, query):
    """
    Извлечь материал по знаку, обработать LLM: убрать лишнее,
    сгладить переходы, сохранить стиль автора. Сохранить в DOCX.
    """
    from pathlib import Path

    import httpx
    from docx import Document
    from docx.shared import Pt

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
    chunk_store = hipporag.chunk_embedding_store
    n_chunks = len(chunk_store.texts)
    q_lower = query.lower()
    sign = _sign_in_query(q_lower)
    if not sign:
        return [("Не указан знак зодиака в запросе", 0, "", 0, 0, 0)]

    # Собираем чанки
    sign_chunks = []
    for idx in range(n_chunks):
        tl = chunk_store.texts[idx].lower()
        if _count_sign(tl, sign) > 0:
            source, page = _get_chunk_meta(chunk_store, idx)
            sign_chunks.append((idx, source, page, chunk_store.texts[idx]))

    if not sign_chunks:
        return [(f"Материал по знаку '{sign}' не найден", 0, "", 0, 0, 0)]

    sign_chunks.sort(key=lambda x: (x[1], x[2], x[0]))
    full_text = "\n\n".join(c[3] for c in sign_chunks)
    total_chunks = len(sign_chunks)

    # Проверяем LLM
    print(f"  📄 Собрано {total_chunks} чанков, {len(full_text)} символов")
    polished = None
    try:
        print("  🔍 Проверка LM Studio...")
        r = httpx.get(f"{api_url}/v1/models", timeout=3)
        if r.status_code == 200:
            # Если текст большой — обрабатываем по частям
            MAX_CHUNK = 3000
            text_parts = []
            remaining = full_text
            while remaining:
                text_parts.append(remaining[:MAX_CHUNK])
                remaining = remaining[MAX_CHUNK:]

            print(f"  🤖 LLM-обработка: {len(text_parts)} частей по {MAX_CHUNK} символов")
            polished_parts = []
            for i, part in enumerate(text_parts):
                print(f"     Часть {i+1}/{len(text_parts)}...", end="", flush=True)
                prompt = {
                    "model": config.LLM_NAME,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Ты — литературный редактор книги Ирины Максимовой «Астрофэншуй квартиры». "
                                f"Твоя задача: обработать отрывок текста о знаке зодиака «{sign}».\n\n"
                                "Правила обработки:\n"
                                "1. Удали информацию, НЕ относящуюся к этому знаку зодиака\n"
                                "2. Сгладь переходы между фрагментами — добавь связующие предложения "
                                "в стиле автора (Ирина Максимова пишет живо, с примерами из практики)\n"
                                "3. Сохрани авторский стиль: личные наблюдения, примеры из практики, "
                                "астрологические термины, фэншуй термины\n"
                                "4. НЕ меняй факты, рекомендации, названия планет и зон\n"
                                "5. Если отрывок начинается с середины мысли — коротко восстанови контекст "
                                "(1-2 предложения в стиле автора)\n"
                                "6. Сохрани все практические советы и нумерацию шагов ремонта\n\n"
                                "Верни ТОЛЬКО обработанный текст, без комментариев."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Часть {i+1} из {len(text_parts)} материала про знак {sign}:\n\n{part}"
                            ),
                        },
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.2,
                }
                resp = httpx.post(f"{api_url}/v1/chat/completions", json=prompt, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    content = (msg.get("content") or "").strip()
                    if not content and msg.get("reasoning_content"):
                        rc = msg["reasoning_content"]
                        lines = [l.strip() for l in rc.split("\n") if l.strip()]
                        content = lines[-1].strip("\"'").replace("**", "") if lines else ""
                    polished_parts.append(content)
                    print(" ✓")
                else:
                    print(f" ⚠️ LLM ответил {resp.status_code}, использую原文")
                    polished_parts.append(part)

            if polished_parts:
                polished = "\n\n".join(polished_parts)
                print(f"  ✅ Обработка завершена: {len(polished)} символов")
    except Exception as e:
        print(f"  ⚠️ Ошибка LLM: {e}")

    if not polished:
        return [
            ("LLM недоступен. Сохранён необработанный материал.", 0, sign, 0, 0, 0)
        ] + extract_sign_full(hipporag, query)

    # Сохраняем DOCX
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{sign}_polished.docx"

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)
    doc.add_heading(f"{sign.upper()} — Астрофэншуй квартиры", level=1)
    doc.add_paragraph("Обработано с сохранением стиля автора")
    doc.add_paragraph("")

    for paragraph in polished.split("\n\n"):
        p = paragraph.strip()
        if not p:
            continue
        if p.isupper() and len(p) < 60:
            doc.add_heading(p, level=2)
        else:
            doc.add_paragraph(p)

    doc.save(str(fname))

    result_text = (
        f"✅ Материал обработан и сохранён\n"
        f">>> Файл: {fname} <<<\n\n"
        f"--- Начало материала ---\n{polished[:2000]}"
    )

    return [(result_text, total_chunks, sign, len(polished), 0, 0)]


def extract_sign_summary(hipporag, query):
    """
    Извлечь материал по знаку, сохранить в DOCX + краткое резюме через LLM.
    """
    from pathlib import Path

    import httpx
    from docx import Document
    from docx.shared import Pt

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL

    # 1. Получаем все чанки (как в extract_sign_full, но без сохранения)
    chunk_store = hipporag.chunk_embedding_store
    n_chunks = len(chunk_store.texts)
    q_lower = query.lower()
    sign = _sign_in_query(q_lower)
    if not sign:
        return [("Не указан знак зодиака в запросе", 0, "", 0, 0, 0)]

    sign_chunks = []
    for idx in range(n_chunks):
        tl = chunk_store.texts[idx].lower()
        if _count_sign(tl, sign) > 0:
            source, page = _get_chunk_meta(chunk_store, idx)
            sign_chunks.append((idx, source, page, chunk_store.texts[idx]))

    if not sign_chunks:
        return [(f"Материал по знаку '{sign}' не найден", 0, "", 0, 0, 0)]

    sign_chunks.sort(key=lambda x: (x[1], x[2], x[0]))
    full_text = "\n".join(c[3] for c in sign_chunks)
    total_chars = len(full_text)
    total_chunks = len(sign_chunks)

    # 2. Сохраняем DOCX
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{sign}.docx"

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)
    doc.add_heading(f"Материал по знаку: {sign.upper()}", level=1)
    doc.add_paragraph("Источник: Ирина Максимова «Астрофэншуй квартиры»")
    doc.add_paragraph(f"Чанков: {total_chunks}, символов: {total_chars}")
    doc.add_paragraph("")
    prev_source = None
    for idx, source, page, text in sign_chunks:
        if prev_source and source != prev_source:
            doc.add_page_break()
            doc.add_heading(f"{source}", level=2)
        doc.add_paragraph(text)
        prev_source = source
    doc.save(str(fname))

    # 3. Резюме через LLM
    input_text = (
        full_text[:4000] + "\n\n[...]\n\n" + full_text[-4000:]
        if len(full_text) > 8000
        else full_text
    )
    summary = "Резюме недоступно (LLM не отвечает)"
    try:
        r = httpx.get(f"{api_url}/v1/models", timeout=3)
        if r.status_code == 200:
            prompt = {
                "model": config.LLM_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты — эксперт по астрологии и фэншуй. Составь краткое резюме "
                            "по знаку зодиака. Выдели: характеристика, зоны квартиры, планеты, "
                            "рекомендации по ремонту и фэншуй. 2-3 абзаца. Без воды."
                        ),
                    },
                    {"role": "user", "content": f"Материал про знак {sign}:\n\n{input_text}"},
                ],
                "max_tokens": 2048,
                "temperature": 0.2,
            }
            resp = httpx.post(f"{api_url}/v1/chat/completions", json=prompt, timeout=300)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                summary = (msg.get("content") or "").strip()
                if not summary and msg.get("reasoning_content"):
                    rc = msg["reasoning_content"]
                    lines = [l.strip() for l in rc.split("\n") if l.strip()]
                    summary = lines[-1].strip("\"'").replace("**", "") if lines else summary
    except Exception:
        pass

    result_text = (
        f"=== КРАТКОЕ РЕЗЮМЕ: {sign.upper()} ===\n\n"
        f"{summary}\n\n"
        f"=== ПОЛНЫЙ ТЕКСТ ===\n"
        f"Файл: {fname}\n"
        f"Чанков: {total_chunks} | Символов: {total_chars}\n"
        f"Откройте DOCX в Word чтобы увидеть весь материал."
    )

    return [(result_text, total_chunks, sign, total_chars, 0, 0)]


def _zones_in_text(text_lower):
    """Канонические названия зон, упомянутых в тексте (в порядке упоминания)."""
    names = []
    for stem, name in _ZONE_TEXT_STEMS.items():
        if stem in text_lower and name not in names:
            names.append(name)
    return names


def _sign_zone_class(sentence_lower):
    """Классификация связи знак→зона: обитель / экзальтация / изгнание / прочее."""
    if "обител" in sentence_lower or "светск" in sentence_lower or "материальн" in sentence_lower:
        return 0
    if "экзальтац" in sentence_lower:
        return 1
    if "изгнан" in sentence_lower or "падени" in sentence_lower:
        return 2
    return 99


def _find_sign_zones(chunk_store, sign, max_evidence=3):
    """Зоны, которые корпус прямо приписывает знаку (P1-8).

    Сканируется **весь** корпус, а не top-50 sentence-level: на вопрос «какая
    зона подходит для Овна?» нужное предложение («обитель овна — марс — это
    плита…») может не попасть в кандидаты, потому что слова вопроса в нём
    отсутствуют. Соседнее предложение-продолжение («И соответствует ему
    прихожая и плита») привязывается к знаку из предыдущего предложения.
    """
    sign_re = _SIGN_RE.get(sign) or _word_re(ZODIAC_FORMS.get(sign, [sign]))
    zones = {}
    prev_sign = False
    prev_chunk = None
    for sentence_lower, sentence, chunk_idx in _sentence_index(chunk_store):
        if prev_chunk != chunk_idx:
            prev_sign = False
        has_sign = bool(sign_re.search(sentence_lower))
        matched = _zones_in_text(sentence_lower)
        strong = any(marker in sentence_lower for marker in _SIGN_ZONE_STRONG)
        weak = strong or any(marker in sentence_lower for marker in _SIGN_ZONE_WEAK)
        continuation = (not has_sign) and prev_sign and weak

        if matched and (has_sign or continuation) and weak:
            if has_sign and strong:
                weight = 3
            else:
                weight = 2
            source, page = _get_chunk_meta(chunk_store, chunk_idx)
            for zone in matched:
                entry = zones.setdefault(zone, {"zone": zone, "weight": 0, "evidence": []})
                entry["weight"] = max(entry["weight"], weight)
                if len(entry["evidence"]) >= max_evidence:
                    continue
                if any(sentence == item[0] for item in entry["evidence"]):
                    continue
                entry["evidence"].append((sentence, source, page, weight))

        prev_sign = has_sign
        prev_chunk = chunk_idx

    return sorted(zones.values(), key=lambda z: (-z["weight"], -len(z["evidence"])))


def _find_zone_signs(chunk_store, zone, sign=None):
    """Пары «знак → управитель» для указанной зоны (P1-8).

    Раньше метод искал regex'ом «знак — … зона» только среди top-50
    предложений и требовал тире сразу после знака. Теперь сканируется весь
    корпус по маркерам обители/экзальтации/изгнания.
    """
    zone_stems = [stem for stem, name in _ZONE_TEXT_STEMS.items() if name == zone]
    if not zone_stems:
        return []
    sign_re = _SIGN_RE
    found = []
    for sentence_lower, sentence, chunk_idx in _sentence_index(chunk_store):
        if not any(stem in sentence_lower for stem in zone_stems):
            continue
        for candidate in ZODIAC_SIGNS:
            if sign and candidate != sign:
                continue
            if not sign_re[candidate].search(sentence_lower):
                continue
            source, page = _get_chunk_meta(chunk_store, chunk_idx)
            found.append(
                {
                    "sign": candidate.capitalize(),
                    "planet": SIGN_PLANET.get(candidate, "?"),
                    "zone": zone,
                    "context": sentence[:300],
                    "source": source,
                    "page": page,
                    "class": _sign_zone_class(sentence_lower),
                }
            )
            break
    return found


def _sign_zone_answer(query, chunk_store, sign):
    """Ответ на вопрос «какая зона подходит для <знака>?» (P1-8)."""
    zones = _find_sign_zones(chunk_store, sign)
    sign_name = sign.capitalize()
    if not zones:
        message = (
            f"Ответ не найден: в источниках нет пары «{sign_name} → зона». "
            "Метод не подменяет данные догадками."
        )
        return [(message, 0, 0, 0, 0)]

    top = zones[:3]
    lines = ["=== РЕЗУЛЬТАТ ПОИСКА ===", f"Запрос: {query}", f"Знак: {sign_name}", ""]
    for zone in top:
        evidence, source, page, _weight = zone["evidence"][0]
        lines.append(f"--- {zone['zone']} ---")
        lines.append(f"[Источник — {source} стр.{page}]")
        lines.append(f"> {evidence[:300]}")
        lines.append("")

    zone_names = ", ".join(zone["zone"] for zone in top)
    lines.append(
        f"ОТВЕТ: Знаку {sign_name} соответствуют зоны: {zone_names}. "
        f"Управитель — {SIGN_PLANET.get(sign, '?')}."
    )
    lines.append("\nМетод: Pattern Search (программный, без LLM)")
    return [("\n".join(lines), 999.0, 0, 0, 0)]


def search_semantic_filter(hipporag, query, top_k=50):
    """
    Метод 22: Поиск по паттерну ЗНАК — ЗОНА + планета-управитель.

    Понимает обе стороны вопроса (P1-8):
    - «какая планета отвечает за плиту?» — зона из запроса;
    - «какая зона подходит для Овна?» — знак из запроса (раньше такой запрос
      давал «Целевая зона: ?» и шум);
    - если в запросе нет ни знака, ни зоны — честный отказ вместо шума.
    """
    import time as _time

    t0 = _time.time()
    print("  Метод 22: Pattern Search (ЗНАК — ЗОНА)", flush=True)

    ql = query.lower()
    target_zone = next((zone for stem, zone in ZONE_ALIASES.items() if stem in ql), None)
    target_sign = _sign_in_query(ql)
    print(
        f"  Зона: {target_zone or 'не определена'} | " f"Знак: {target_sign or 'не определён'}",
        flush=True,
    )

    if not target_zone and not target_sign:
        message = (
            "Ответ не найден: метод ищет пары «знак → зона». Укажите в вопросе "
            "знак зодиака (например «Овен») или зону (например «плита», «спальня»)."
        )
        return [(message, 0, 0, 0, 0)]

    chunk_store = hipporag.chunk_embedding_store

    if target_sign and not target_zone:
        result = _sign_zone_answer(query, chunk_store, target_sign)
        t1 = _time.time()
        print(f"  Готово за {t1 - t0:.1f}с", flush=True)
        return result

    found = _find_zone_signs(chunk_store, target_zone, sign=target_sign)
    print(f"  Найдено соответствий: {len(found)}", flush=True)

    if not found:
        message = (
            f"Ответ не найден: в источниках нет соответствия для «{target_zone}»"
            + (f" и знака {target_sign.capitalize()}" if target_sign else "")
            + "."
        )
        return [(message, 0, 0, 0, 0)]

    # Лучшее совпадение из каждой классификации
    found.sort(key=lambda item: (item["class"], item["page"]))
    best_per_class = {}
    for item in found:
        best_per_class.setdefault(item["class"], item)

    zone_acc = _ACCUSATIVE.get(target_zone, target_zone)
    lines = [
        "=== РЕЗУЛЬТАТ ПОИСКА ===",
        f"Запрос: {query}",
        f"Целевая зона: {target_zone}",
    ]
    if target_sign:
        lines.append(f"Знак: {target_sign.capitalize()}")
    lines.append("")

    primary = best_per_class.get(0) or next(iter(best_per_class.values()))
    primary_name = _CLASS_NAMES.get(primary["class"], "прочее соответствие")
    lines.append(f"Основное соответствие ({primary_name}):")
    lines.append(f"[Источник — {primary['source']} стр.{primary['page']}]")
    lines.append(f"> {primary['context'][:300]}")
    lines.append(f"Знак {primary['sign']} → управитель {primary['planet']}")
    lines.append(f"ОТВЕТ: За зону {zone_acc} отвечает планета {primary['planet']}.")

    others = [item for item in best_per_class.values() if item is not primary]
    if others:
        lines.append("")
        lines.append("--- Другие классификации ---")
        for item in others:
            class_name = _CLASS_NAMES.get(item["class"], "прочее")
            lines.append(
                f"[{class_name}] {item['sign']} → {item['planet']} "
                f"({item['source']} стр.{item['page']})"
            )

    lines.append("\nМетод: Pattern Search (программный, без LLM)")
    t1 = _time.time()
    print(f"  Готово за {t1 - t0:.1f}с", flush=True)
    return [("\n".join(lines), 999.0, 0, 0, 0)]
