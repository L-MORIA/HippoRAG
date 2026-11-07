"""Обработка запроса: query expansion, переформулировка, маркеры метазапросов.

Перенос из main.py (шаг 4 миграции MAIN_SPLIT_PLAN.md) — тела функций извлечены
байт-в-байт (417-662); изменена только секция импортов:
SYNONYM_MAP -> load_synonyms() из .synonyms (данные пакета), STOP_WORDS/config.LLM_NAME
-> .config.
"""

import re

import httpx

from . import config
from .config import LLM_BASE_URL, STOP_WORDS
from .synonyms import load_synonyms

SYNONYM_MAP = load_synonyms()


def _expand_query(q_lower, terms):
    """Расширить запрос синонимами из словаря.
    Автоматически нормализует термины: убирает пунктуацию, пробует падежные формы."""
    import re as _re

    expanded = set(terms)
    for term in terms:
        # Пробуем исходный термин
        candidates = [term]
        # Убираем пунктуацию
        clean = _re.sub(r"[^\w\s]", "", term)
        if clean != term and len(clean) > 2:
            candidates.append(clean)
        for cand in candidates:
            if cand in SYNONYM_MAP:
                for syn in SYNONYM_MAP[cand]:
                    if len(syn) > 2 and syn not in expanded:
                        expanded.add(syn)
                break
    # Числовой мост «гуа 5 → пятёрка» (уточнение автора книг, 2026-11-07,
    # docs/AUTHOR_CLARIFICATION_GUA5_2026-11-07.md). В книге направления
    # пятого числа сформулированы словом «пятёрка» без «гуа 5» («Мужчина-
    # пятёрка опирается на знания (северо-восток)...», стр. 25), поэтому
    # словарные синонимы их не находят. Мост срабатывает только при явном
    # «гуа» и отдельной цифре 5 в запросе — остальные вопросы про гуа
    # он не затрагивает.
    if _re.search(r"гуа", q_lower) and _re.search(r"(?<!\d)5(?!\d)", q_lower):
        expanded.update(("пятёрка", "пятёрки", "пятёрок"))
    return list(expanded)


def _word_re(words):
    """Regex, совпадающий с любым из слов целиком (без подстрок).

    Нужен для знаков: «рак» — подстрока «брак», «дева» — «девальвация».
    Без границ слова «какой брак?» определялся как знак Рак (P1-8).
    """
    ordered = sorted({w for w in words if w}, key=len, reverse=True)
    return re.compile(
        r"(?<![а-яёa-z0-9])(" + "|".join(re.escape(w) for w in ordered) + r")(?![а-яёa-z0-9])"
    )


# Знаки зодиака + все падежные формы (для поиска по любому падежу)
ZODIAC_SIGNS = [
    "овен",
    "телец",
    "близнецы",
    "рак",
    "лев",
    "дева",
    "весы",
    "скорпион",
    "стрелец",
    "козерог",
    "водолей",
    "рыбы",
]
# Все возможные падежные формы знаков зодиака
ZODIAC_FORMS = {
    "овен": ["овен", "овна", "овну", "овном", "овне", "овны", "овнов"],
    "телец": [
        "телец",
        "тельца",
        "тельцу",
        "тельцом",
        "тельце",
        "тельцы",
        "тельцов",
        "тельцам",
        "тельцами",
        "тельцах",
    ],
    "близнецы": ["близнецы", "близнецов", "близнецам", "близнецами", "близнецах"],
    "рак": ["рак", "рака", "раку", "раком", "раке", "раки", "раков"],
    "лев": ["лев", "льва", "льву", "львом", "льве", "львы", "львов"],
    "дева": ["дева", "девы", "деве", "деву", "девой", "дев"],
    "весы": ["весы", "весов", "весам", "весами", "весах"],
    "скорпион": [
        "скорпион",
        "скорпиона",
        "скорпиону",
        "скорпионом",
        "скорпионе",
        "скорпионы",
        "скорпионов",
    ],
    "стрелец": [
        "стрелец",
        "стрельца",
        "стрельцу",
        "стрельцом",
        "стрельце",
        "стрельцы",
        "стрельцов",
    ],
    "козерог": [
        "козерог",
        "козерога",
        "козерогу",
        "козерогом",
        "козероге",
        "козероги",
        "козерогов",
    ],
    "водолей": ["водолей", "водолея", "водолею", "водолеем", "водолее", "водолеи", "водолеев"],
    "рыбы": ["рыбы", "рыб", "рыбам", "рыбами", "рыбах"],
}
# Строим маппинг: любая форма → именительный падеж
FORM_TO_SIGN = {}
for nominative, forms in ZODIAC_FORMS.items():
    for f in forms:
        FORM_TO_SIGN[f] = nominative

# Один скомпилированный regex на знак: используется в горячих циклах
# (_sign_in_query, _find_sign_zones, _find_zone_signs).
_SIGN_RE = {sign: _word_re(forms) for sign, forms in ZODIAC_FORMS.items()}
_ALL_SIGN_FORMS_RE = _word_re(FORM_TO_SIGN)

# Планеты и ключевые термины для расширенного анализа
ASTRO_TERMS = ZODIAC_SIGNS + [
    "марс",
    "венера",
    "меркурий",
    "юпитер",
    "сатурн",
    "уран",
    "нептун",
    "плутон",
    "луна",
    "солнце",
    "асцендент",
    "десцендент",
    "фэншуй",
    "багуа",
]


_META_REFORMULATION_MARKERS = (
    "не содержит",
    "нет астрологических",
    "невозможно улучшить",
    "нельзя улучшить",
    "запрос не",
    "улучшить запрос",
    "уже хорош",
    "не требуется",
    "без изменений",
    "does not contain",
    "cannot be improved",
    "no improvement",
)


def _is_meta_reformulation(content: str) -> bool:
    """Ответ LLM — мета-текст («Запрос не содержит…»), а не поисковый запрос.

    Такой ответ нельзя отдавать в эмбеддинг: его термины не совпадают ни с
    одним чанком, keyword-boost обнуляется, и выдача перестаёт релевантной
    (дефект режима 1, измерен 2026-11-02).
    """
    low = content.lower()
    return any(marker in low for marker in _META_REFORMULATION_MARKERS)


def _llm_timeout() -> int:
    """Единый бюджет ожидания LLM (F-6); LLM_TIMEOUT переопределяет."""
    import os

    try:
        return int(os.getenv("LLM_TIMEOUT", "90"))
    except ValueError:
        return 90


def reformulate_query(hipporag, query):
    """Улучшить запрос через LLM (прямой вызов LM Studio API)."""

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL

    # Проверка: отвечает ли LM Studio
    try:
        r = httpx.get(f"{api_url}/v1/models", timeout=3)
        if r.status_code != 200:
            print(f"\n  ⚠️ LM Studio не отвечает (статус {r.status_code}).")
            print("  ⚠️ Переформулировка запроса через LLM недоступна.")
            print("  ⚠️ Поиск продолжается без переформулировки.\n")
            return query
    except httpx.HTTPError:
        # F-4: httpx.HTTPError покрывает RemoteProtocolError, ReadError,
        # PoolTimeout и пр. — весь класс сетевых сбоев при перезапуске LM
        # Studio. Узкая пара (ConnectError, TimeoutException) пропускала
        # остальные, и они уходили исключением наверх → 500 в режиме 1.
        print("\n  ⚠️ LM Studio не запущен или не отвечает.")
        print("  ⚠️ Переформулировка запроса через LLM невозможна.")
        print("  ⚠️ Чтобы включить — запустите LM Studio и загрузите модель.")
        print("  ⚠️ Поиск продолжается без переформулировки.\n")
        return query

    # Прямой вызов LM Studio API (chat completions)
    try:
        payload = {
            "model": config.LLM_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты — эксперт по астрологии и фэншуй. "
                        "Улучши поисковый запрос: выдели ключевые термины "
                        "(знак зодиака, планета, зона дома, действие). "
                        "Ответь ОДНИМ предложением (3-10 слов). Без пояснений.\n"
                        "ВАЖНО: запрос может быть о любой теме (автор, книга, издание). "
                        "В таком случае выдели ключевые слова запроса (имя, название, тема) "
                        "и верни их как поисковый запрос. "
                        "НИКОГДА не отвечай комментарием о самом запросе "
                        "(например «Запрос не содержит…») — верни только сам запрос."
                    ),
                },
                {"role": "user", "content": f"Запрос: {query}"},
            ],
            "max_tokens": 2048,
            "temperature": 0,
        }
        # F-6: единый бюджет LLM_TIMEOUT (по умолчанию 90 c), а не 300 c
        r = httpx.post(
            f"{api_url}/v1/chat/completions",
            json=payload,
            timeout=_llm_timeout(),
        )
        if r.status_code == 200:
            data = r.json()
            msg = data["choices"][0]["message"]
            finish = data["choices"][0].get("finish_reason", "")

            content = (msg.get("content") or "").strip().strip("\"'")
            # У reasoning-моделей (gemma-4-e4b) ответ в reasoning_content, если content пуст
            if not content and finish == "stop":
                rc = msg.get("reasoning_content", "")
                if rc:
                    lines = [l.strip() for l in rc.split("\n") if l.strip()]
                    content = lines[-1].strip("\"'").replace("**", "") if lines else ""

            if content and len(content) > 5:
                if _is_meta_reformulation(content):
                    print(
                        "  ⚠️ LLM ответил мета-текстом вместо запроса — "
                        "используется оригинальный запрос"
                    )
                    return query
                print(f"  ✅ Переформулировка: '{query}' → '{content}'")
                return content
        print("  ⚠️ LLM вернул пустой ответ, используется оригинальный запрос")
    except Exception as e:
        print(f"\n  ⚠️ Ошибка LLM: {e}")
        print("  ⚠️ Переформулировка запроса через LLM недоступна.")
        print("  ⚠️ Поиск продолжается без переформулировки.\n")
    return query


def split_query_terms(query):
    """Разбить запрос на значимые термы.

    Пунктуация снимается: без этого «Овна?» превращалось в токен 'овна?',
    который не совпадал ни с одним словом корпуса — BM25 и keyword-boost
    теряли совпадения на любом запросе со знаком вопроса (P2-10).
    """
    q = query.lower()
    terms = []
    for w in q.split():
        w = re.sub(r"[^\w\s]", "", w)
        if len(w) > 2 and w not in STOP_WORDS:
            terms.append(w)
    return terms, q
