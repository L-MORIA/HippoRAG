import json
import os
import re
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import min_max_normalize

SAVE_DIR = "outputs/astrology_test"
LLM_NAME = os.getenv("LLM_NAME", "gemma-4-e4b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

logging.basicConfig(level=logging.WARNING)

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

# Словарь синонимов и связанных терминов для Query Expansion
SYNONYM_MAP = {
    # Знаки зодиака → синонимы, латиница, связанные термины
    "овен": [
        "овен",
        "овна",
        "овну",
        "овном",
        "aries",
        "марс",
        "плита",
        "прихожая",
        "первый знак",
        "стихия огня",
    ],
    "телец": [
        "телец",
        "тельца",
        "тельцу",
        "тельцом",
        "taurus",
        "венера",
        "кладовка",
        "шкаф",
        "второй знак",
        "стихия земли",
    ],
    "близнецы": [
        "близнецы",
        "близнецов",
        "gemini",
        "меркурий",
        "коридор",
        "телефон",
        "третий знак",
        "стихия воздуха",
    ],
    "рак": [
        "рак",
        "рака",
        "раку",
        "cancer",
        "луна",
        "ванная",
        "баня",
        "четвертый знак",
        "стихия воды",
    ],
    "лев": ["лев", "льва", "льву", "leo", "солнце", "гостиная", "зал", "пятый знак", "стихия огня"],
    "дева": [
        "дева",
        "девы",
        "virgo",
        "меркурий",
        "кухня",
        "столовая",
        "шестой знак",
        "стихия земли",
    ],
    "весы": ["весы", "весов", "libra", "венера", "коридор общий", "седьмой знак", "стихия воздуха"],
    "скорпион": [
        "скорпион",
        "скорпиона",
        "scorpio",
        "плутон",
        "туалет",
        "подвал",
        "восьмой знак",
        "стихия воды",
        "марс",
    ],
    "стрелец": [
        "стрелец",
        "стрельца",
        "sagittarius",
        "юпитер",
        "балкон",
        "лоджия",
        "девятый знак",
        "стихия огня",
    ],
    "козерог": [
        "козерог",
        "козерога",
        "capricorn",
        "сатурн",
        "прихожая вход",
        "десятый знак",
        "стихия земли",
    ],
    "водолей": [
        "водолей",
        "водолея",
        "aquarius",
        "уран",
        "спальня",
        "одиннадцатый знак",
        "стихия воздуха",
    ],
    "рыбы": [
        "рыбы",
        "рыб",
        "pisces",
        "нептун",
        "кладовая",
        "холодильник",
        "двенадцатый знак",
        "стихия воды",
    ],
    # Планеты → зоны квартиры
    "марс": ["марс", "марса", "плита", "прихожая", "спортзал", "огонь", "энергия", "напор"],
    "венера": [
        "венера",
        "венеры",
        "кладовка",
        "шкаф",
        "холодильник",
        "эстетика",
        "красота",
        "комфорт",
        "деньги",
    ],
    "меркурий": ["меркурий", "меркурия", "коридор", "телефон", "связь", "коммуникация", "общение"],
    "юпитер": [
        "юпитер",
        "юпитера",
        "балкон",
        "лоджия",
        "библиотека",
        "терраса",
        "расширение",
        "удача",
    ],
    "сатурн": [
        "сатурн",
        "сатурна",
        "входная дверь",
        "порог",
        "крыльцо",
        "ограничение",
        "время",
        "структура",
    ],
    "уран": ["уран", "урана", "спальня", "техника", "новшество", "электрика", "нестандартность"],
    "нептун": ["нептун", "нептуна", "кладовая", "подвал", "тайна", "подсознание", "вода"],
    "плутон": ["плутон", "плутона", "туалет", "подвал", "трансформация", "старьё", "экстрим"],
    "луна": ["луна", "луны", "ванная", "баня", "семья", "род", "архив", "эмоции", "привычки"],
    "солнце": ["солнце", "солнца", "гостиная", "зал", "жизнь", "творчество", "успех", "здоровье"],
    # Астрологические дома
    "1 дом": ["первый дом", "1 дом", "асцендент", "внешность", "личность", "я"],
    "2 дом": ["второй дом", "2 дом", "деньги", "имущество", "ценности", "доход"],
    "3 дом": ["третий дом", "3 дом", "общение", "соседи", "поездки", "информация"],
    "4 дом": ["четвертый дом", "4 дом", "ic", "семья", "род", "дом", "родители", "недвижимость"],
    "5 дом": ["пятый дом", "5 дом", "любовь", "дети", "творчество", "хобби", "романтика"],
    "6 дом": ["шестой дом", "6 дом", "здоровье", "работа", "служба", "повседневность"],
    "7 дом": ["седьмой дом", "7 дом", "десцендент", "брак", "партнёр", "отношения"],
    "8 дом": ["восьмой дом", "8 дом", "кризис", "перемены", "секс", "чужие деньги", "наследство"],
    "9 дом": ["девятый дом", "9 дом", "путешествия", "образование", "философия", "религия"],
    "10 дом": ["десятый дом", "10 дом", "мс", "карьера", "статус", "профессия", "цель"],
    "11 дом": ["одиннадцатый дом", "11 дом", "друзья", "единомышленники", "надежды", "проекты"],
    "12 дом": ["двенадцатый дом", "12 дом", "уединение", "отдых", "подсознание", "тайны"],
    # Зоны квартиры (фэншуй) → синонимы + связанные планеты
    "прихожая": [
        "прихожая",
        "прихожей",
        "прихожую",
        "холл",
        "коридор входной",
        "входная зона",
        "порог",
        "марс",
    ],
    "плита": ["плита", "плиты", "плиту", "плитой", "плите", "прихожая", "марс", "овен", "кухня"],
    "кухня": [
        "кухня",
        "кухни",
        "кухню",
        "кухне",
        "кухонный",
        "столовая",
        "место готовки",
        "зона питания",
        "плита",
        "дева",
        "меркурий",
    ],
    "спальня": [
        "спальня",
        "спальни",
        "спальню",
        "спальне",
        "спальный",
        "комната отдыха",
        "место сна",
    ],
    "гостиная": ["гостиная", "гостиной", "гостиную", "зал", "общая комната", "приём гостей"],
    "ванная": ["ванная", "ванной", "ванную", "ванна", "душ", "санузел", "вода"],
    "туалет": ["туалет", "туалета", "туалету", "сортир", "уборная", "санузел"],
    "коридор": [
        "коридор",
        "коридора",
        "коридору",
        "коридором",
        "проход",
        "анфилада",
        "соединительная зона",
    ],
    "кладовка": [
        "кладовка",
        "кладовки",
        "кладовку",
        "кладовке",
        "кладовая",
        "шкаф",
        "хранение",
        "венера",
        "телец",
    ],
    "шкаф": [
        "шкаф",
        "шкафа",
        "шкафу",
        "шкафом",
        "шкафы",
        "кладовка",
        "хранение",
        "венера",
        "телец",
    ],
    "балкон": ["балкон", "балкона", "балкону", "балконом", "лоджия", "терраса"],
    "дверь": ["дверь", "двери", "дверью", "вход", "входная дверь"],
    # Обратные связи: падежные формы → базовые слова
    "плиту": ["плита", "плиты", "плитой", "прихожая", "марс", "овен", "кухня"],
    "плиты": ["плита", "плиту", "плитой", "прихожая", "марс", "овен", "кухня"],
    "плите": ["плита", "плиты", "плиту", "прихожая", "марс", "овен", "кухня"],
    "прихожей": ["прихожая", "холл", "коридор входной", "марс"],
    "прихожую": ["прихожая", "холл", "коридор входной", "марс"],
    "кладовки": ["кладовка", "кладовая", "шкаф", "хранение", "венера", "телец"],
    "кладовку": ["кладовка", "кладовая", "шкаф", "хранение", "венера", "телец"],
    "шкафа": ["шкаф", "шкафу", "шкафом", "шкафы", "кладовка", "хранение", "венера", "телец"],
    "шкафу": ["шкаф", "шкафа", "шкафом", "шкафы", "кладовка", "хранение", "венера", "телец"],
    # Планета → все планеты (для запросов "какая планета отвечает за...")
    "планета": [
        "планета",
        "планеты",
        "планету",
        "планетой",
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
    ],
    "планеты": [
        "планета",
        "планету",
        "планетой",
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
    ],
    "планету": [
        "планета",
        "планеты",
        "планетой",
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
    ],
    # Ключевые понятия
    "ремонт": [
        "ремонт",
        "ремонта",
        "ремонту",
        "ремонте",
        "ремонтный",
        "отделка",
        "перепланировка",
        "обновление",
        "строительство",
        "преображение",
        "шаги ремонта",
        "этапы ремонта",
    ],
    "ремонта": [
        "ремонт",
        "ремонту",
        "ремонте",
        "ремонтный",
        "отделка",
        "перепланировка",
        "шаги ремонта",
    ],
    "ремонту": ["ремонт", "ремонта", "ремонте", "отделка", "перепланировка"],
    "ремонте": ["ремонт", "ремонта", "ремонту", "отделка", "перепланировка"],
    "фэншуй": [
        "фэншуй",
        "фэн-шуй",
        "феншуй",
        "багуа",
        "ци",
        "гармония",
        "баланс",
        "энергия дома",
        "организация пространства",
    ],
    "захламление": [
        "захламление",
        "хлам",
        "старьё",
        "мусор",
        "беспорядок",
        "барахло",
        "расхламление",
    ],
    "интерьер": [
        "интерьер",
        "интерьера",
        "интерьеру",
        "дизайн",
        "оформление",
        "обстановка",
        "убранство",
        "стиль",
    ],
    "астрология": [
        "астрология",
        "астрологии",
        "астрофэншуй",
        "гороскоп",
        "натальная карта",
        "зодиак",
    ],
}


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


def reformulate_query(hipporag, query):
    """Улучшить запрос через LLM (прямой вызов LM Studio API)."""
    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL

    # Проверка: отвечает ли LM Studio
    try:
        r = httpx.get(f"{api_url}/v1/models", timeout=3)
        if r.status_code != 200:
            print(f"\n  ⚠️ LM Studio не отвечает (статус {r.status_code}).")
            print("  ⚠️ Переформулировка запроса через LLM недоступна.")
            print("  ⚠️ Поиск продолжается без переформулировки.\n")
            return query
    except (httpx.ConnectError, httpx.TimeoutException):
        print("\n  ⚠️ LM Studio не запущен или не отвечает.")
        print("  ⚠️ Переформулировка запроса через LLM невозможна.")
        print("  ⚠️ Чтобы включить — запустите LM Studio и загрузите модель.")
        print("  ⚠️ Поиск продолжается без переформулировки.\n")
        return query

    # Прямой вызов LM Studio API (chat completions)
    try:
        payload = {
            "model": LLM_NAME,
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
        r = httpx.post(f"{api_url}/v1/chat/completions", json=payload, timeout=300)
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


def search_multi_vector(hipporag, query, top_k=10, candidate_pool=200):
    """
    Multi-vector поиск:
    1. Переформулировка запроса через LLM
    2. Полный запрос → эмбеддинг + similarity
    3. Каждый терм запроса → эмбеддинг + similarity (max-pooling)
    4. Большой keyword boost (1.0)
    5. Комбинированный скор
    """
    # Шаг 0: Переформулировка
    search_query = reformulate_query(hipporag, query)

    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        print("No chunk embeddings found.")
        return []

    all_emb = np.array(chunk_store.embeddings)
    n_chunks = len(all_emb)

    # Шаг 1: эмбеддинг полного запроса
    full_emb = hipporag.embedding_model.encode([search_query])
    full_scores = np.dot(all_emb, full_emb.T).squeeze()

    # Шаг 2: эмбеддинг каждого терма
    terms, q_lower = split_query_terms(search_query)
    if not terms:
        terms = [w for w in search_query.lower().split() if len(w) > 2]

    multi_scores = np.zeros(n_chunks)
    if terms:
        term_embs = hipporag.embedding_model.encode(terms)
        for te in term_embs:
            ts = np.dot(all_emb, te.reshape(-1, 1)).squeeze()
            multi_scores = np.maximum(multi_scores, ts)  # max-pooling

    # Шаг 3: сортировка по full_scores для candidate pool
    sorted_idx = np.argsort(full_scores)[::-1]
    pool = min(candidate_pool, n_chunks)

    scored = []
    for i in range(pool):
        idx = sorted_idx[i]
        text = chunk_store.texts[idx]
        tl = text.lower()

        dpr = float(full_scores[idx])
        multi = float(multi_scores[idx]) if terms else 0.0

        # Keyword boost
        boost = 0.0
        if q_lower in tl:
            boost += 1.0
        if terms:
            matches = sum(1 for t in terms if t in tl)
            boost += 0.3 * min(matches, 5) / max(len(terms), 1)

        # Комбинируем: full-query (40%) + multi-vector (60%) + keyword boost
        combo = dpr * 0.4 + multi * 0.6
        final = combo * (1.0 + boost)
        scored.append((text, final, dpr, multi, boost))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k], search_query


def search_hipporag_graph(hipporag, query, top_k=10, linking_entities=30, passage_weight=0.3):
    """Графовый поиск: entity linking + PPR на графе."""
    hipporag.prepare_retrieval_objects()

    # Определяем знак зодиака для точного entity linking
    _, q_lower = split_query_terms(query)
    terms, _ = split_query_terms(query)

    q_emb = hipporag.embedding_model.encode([query])
    entity_scores = np.dot(hipporag.entity_embeddings, q_emb.T).squeeze()

    # Sign dominance boost для entity scoring
    sign_in_query = _sign_in_query(q_lower)
    if sign_in_query:
        for i, e_text in enumerate(hipporag.entity_embedding_store.texts):
            if sign_in_query in str(e_text).lower():
                entity_scores[i] *= 1.5

    node_weights = np.zeros(hipporag.graph.vcount())
    sorted_entity_idx = np.argsort(entity_scores)[::-1][:linking_entities]
    for e_idx in sorted_entity_idx:
        vid = hipporag.entity_node_idxs[e_idx]
        node_weights[vid] = entity_scores[e_idx]

    dpr_ids, dpr_scores = hipporag.dense_passage_retrieval(query)
    norm_dpr = min_max_normalize(dpr_scores)
    for i, pid in enumerate(dpr_ids):
        pkey = hipporag.passage_node_keys[pid]
        if pkey in hipporag.node_name_to_vertex_idx:
            vid = hipporag.node_name_to_vertex_idx[pkey]
            node_weights[vid] += norm_dpr[i] * passage_weight

    if sum(node_weights) == 0:
        ppr_ids, ppr_scores = dpr_ids, dpr_scores
    else:
        ppr_ids, ppr_scores = hipporag.run_ppr(node_weights, damping=0.3)

    chunk_store = hipporag.chunk_embedding_store
    scored = []
    for i in range(min(len(ppr_ids), len(dpr_ids))):
        idx = ppr_ids[i]
        text = chunk_store.texts[idx]
        ppr_score = float(ppr_scores[i])
        dpr_idx = np.where(dpr_ids == idx)[0]
        dpr_score = float(dpr_scores[dpr_idx[0]]) if len(dpr_idx) > 0 else 0.0
        tl = text.lower()
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final_score = ppr_score * (1.0 + boost)
        scored.append((text, final_score, ppr_score, dpr_score, boost))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _tokenize(text):
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def _build_bm25(chunk_store):
    # Кэш привязан к конкретному хранилищу: раньше он был глобальным и при
    # смене индекса (или в тестах) молча возвращал BM25 от другого корпуса.
    cache_key = (id(chunk_store), len(chunk_store.texts))
    if getattr(_build_bm25, "cache_key", None) != cache_key:
        corpus = [_tokenize(t) for t in chunk_store.texts]
        _build_bm25.cache = BM25Okapi(corpus)
        _build_bm25.cache_key = cache_key
    return _build_bm25.cache


def _embedding_scores(hipporag, chunk_store, query):
    """Косинусные скоры запроса по всем чанкам (dense-канал).

    Возвращает None, если эмбеддинги недоступны — вызывающий код тогда
    остаётся на BM25. Нужно для Self-RAG (P1-6): чистый BM25 не находит
    титульные страницы, потому что там нет слов вопроса.
    """
    embeddings = getattr(chunk_store, "embeddings", None)
    if embeddings is None or len(embeddings) == 0:
        return None
    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.asarray(embeddings)
    return np.atleast_1d(np.asarray(np.dot(all_emb, q_emb.T)).squeeze())


_CROSS_ENCODER = None
_CROSS_ENCODER_RU = None


def _get_crossencoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER


def _get_crossencoder_ru():
    global _CROSS_ENCODER_RU
    if _CROSS_ENCODER_RU is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER_RU = CrossEncoder("DiTy/cross-encoder-russian-msmarco")
    return _CROSS_ENCODER_RU


def _rerank_with_crossencoder_ru(query, candidates):
    if not candidates:
        return candidates
    try:
        ce = _get_crossencoder_ru()
        pairs = [(query, c[0][:512]) for c in candidates]
        xc_scores = ce.predict(pairs, show_progress_bar=False)
        # Сырой CE-скор у RU-модели — вероятность (0…1), у EN — логит (…±11),
        # а old — embedding-скор (0…1). Смешивать их напрямую бессмысленно:
        # коэффициент 0.7/0.3 не означает ничего. Нормируем xc (P1-4).
        xc_scores = min_max_normalize(np.array([float(x) for x in xc_scores]))
        reranked = []
        for i, c in enumerate(candidates):
            xc = float(xc_scores[i])
            text = c[0]
            old = c[1]
            boost = c[4] if len(c) > 4 else 0.0
            final = xc * 0.7 + old * 0.3 * (1.0 + boost)
            reranked.append((text, final, xc, old, boost))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    except Exception as e:
        print(f"  \u26a0\ufe0f Русский Cross-encoder недоступен ({e})")
        return candidates


def _rerank_with_crossencoder(query, candidates):
    if not candidates:
        return candidates
    try:
        ce = _get_crossencoder()
        pairs = [(query, c[0][:512]) for c in candidates]
        xc_scores = ce.predict(pairs, show_progress_bar=False)
        # Нормировка CE-скора перед смешиванием — см. P1-4 (в EN-ветке сырой
        # логит давал 86 % веса в итоговом скоре и переставлял кандидатов).
        xc_scores = min_max_normalize(np.array([float(x) for x in xc_scores]))
        reranked = []
        for i, c in enumerate(candidates):
            xc = float(xc_scores[i])
            text = c[0]
            old = c[1]
            boost = c[4] if len(c) > 4 else 0.0
            final = xc * 0.7 + old * 0.3 * (1.0 + boost)
            reranked.append((text, final, xc, old, boost))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    except Exception as e:
        print(f"  \u26a0\ufe0f Cross-encoder недоступен ({e}), используется исходный порядок")
        return candidates


_E5_MODEL = None


def search_mxbai(hipporag, query, top_k=10, candidate_pool=200):
    """
    Поиск через BM25 + существующие эмбеддинги (RRF) + sign dominance.
    mxbai-embed и mxbai-rerank — английские модели, не работают для русского домена.
    Метод использует проверенную RRF-комбинацию (как метод 13).
    """
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    expanded_terms = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + expanded_terms))
    bm25_scores = bm25.get_scores(q_tokens)

    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()

    # RRF
    bm25_rank = np.argsort(bm25_scores)[::-1]
    emb_rank = np.argsort(emb_scores)[::-1]
    bm25_pos = {idx: pos for pos, idx in enumerate(bm25_rank)}
    emb_pos = {idx: pos for pos, idx in enumerate(emb_rank)}
    k = 60
    rrf_scores = [
        1 / (k + bm25_pos.get(i, 1e6)) + 1 / (k + emb_pos.get(i, 1e6)) for i in range(n_chunks)
    ]
    candidate_idx = np.argsort(rrf_scores)[::-1][:candidate_pool]
    if len(candidate_idx) == 0:
        return []

    print("  ✅ RRF (BM25 + эмбеддинги) + sign dominance + контекст", flush=True)

    scored = []
    for idx in candidate_idx:
        text = chunk_store.texts[idx]
        tl = text.lower()
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        emb = float(emb_scores[idx])
        rrf = float(rrf_scores[idx])
        final = emb * 0.6 + rrf * 0.4 * (1.0 + boost)
        scored.append((idx, final, emb, rrf, boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Контекстное окно ±1
    used_ranges = set()
    merged = []
    for idx, final, emb_sim, rrf, boost in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-1, 2):
            ni = idx + offset
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= 1:
                window.add(ni)
        pages = set()
        for wi in window:
            _, wp = _get_chunk_meta(chunk_store, wi)
            pages.add(wp)
        min_p, max_p = min(pages), max(pages)
        rkey = f"{source}:{min_p}-{max_p}"
        if rkey in used_ranges:
            continue
        used_ranges.add(rkey)
        parts = [chunk_store.texts[wi] for wi in sorted(window)]
        full_text = "\n\n".join(parts)
        boost_final = _keyword_boost(full_text.lower(), q_lower, terms)
        final_score = emb_sim * (1.0 + boost_final)
        merged.append((full_text, final_score, source, min_p, max_p, boost_final))
        if len(merged) >= top_k:
            break

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def _get_e5():
    global _E5_MODEL
    if _E5_MODEL is None:
        from sentence_transformers import SentenceTransformer

        print("  ⏳ Загрузка E5 модели...", flush=True)
        _E5_MODEL = SentenceTransformer("intfloat/multilingual-e5-small")
        print("  ✅ E5 модель загружена", flush=True)
    return _E5_MODEL


def search_e5_rerank(hipporag, query, top_k=10, candidate_pool=200):
    """
    Поиск: существующие эмбеддинги (1369 чанков) → E5 sentence-rerank top 50.
    E5 эмбеддит только query + предложения топ-50 кандидатов (не все чанки).
    Качество не падает — BM25 + существующие эмбеддинги отбирают лучших кандидатов,
    E5 только доуточняет ранжирование по предложениям.
    """
    import re as _re

    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    expanded_terms = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + expanded_terms))
    bm25_scores = bm25.get_scores(q_tokens)

    # Кандидаты: комбинация BM25 + существующие эмбеддинги (все 1369 чанков)
    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()
    bm25_rank = np.argsort(bm25_scores)[::-1]
    emb_rank = np.argsort(emb_scores)[::-1]
    bm25_pos = {idx: pos for pos, idx in enumerate(bm25_rank)}
    emb_pos = {idx: pos for pos, idx in enumerate(emb_rank)}
    k = 60
    rrf_scores = [
        1 / (k + bm25_pos.get(i, 1e6)) + 1 / (k + emb_pos.get(i, 1e6)) for i in range(n_chunks)
    ]
    candidate_idx = np.argsort(rrf_scores)[::-1][:candidate_pool]

    if len(candidate_idx) == 0:
        return []

    # E5: только для sentence-reranking топ-50
    e5 = _get_e5()
    q_sentences = [s.strip() for s in _re.split(r"[.!?]+", query) if len(s.strip()) > 3]
    if not q_sentences:
        q_sentences = [query]
    q_sent_embs = e5.encode([f"query: {s}" for s in q_sentences])

    top50 = candidate_idx[:50]
    rest = candidate_idx[50:]

    scored = []
    for idx in top50:
        text = chunk_store.texts[idx]
        tl = text.lower()

        # Sentence-level через E5
        chunk_sents = [s.strip() for s in _re.split(r"[.!?]+", text) if len(s.strip()) > 10]
        sent_score = 0
        if chunk_sents:
            chunk_sents = chunk_sents[:10]
            cs_embs = e5.encode([f"passage: {s}" for s in chunk_sents])
            sim = np.dot(cs_embs, q_sent_embs.T)
            sent_score = float(np.max(sim))

        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score

        final = (
            float(emb_scores[idx]) * 0.4
            + sent_score * 0.4
            + float(emb_scores[idx]) * 0.2 * max(boost, 0)
        )
        scored.append((idx, final, float(emb_scores[idx]), sent_score, boost))

    for idx in rest:
        text = chunk_store.texts[idx]
        tl = text.lower()
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final = float(emb_scores[idx]) * (1.0 + boost)
        scored.append((idx, final, float(emb_scores[idx]), 0, boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Контекстное окно ±1
    used_ranges = set()
    merged = []
    for idx, final, emb_sim, sent_score, boost in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-1, 2):
            ni = idx + offset
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= 1:
                window.add(ni)
        pages = set()
        for wi in window:
            _, wp = _get_chunk_meta(chunk_store, wi)
            pages.add(wp)
        min_p, max_p = min(pages), max(pages)
        rkey = f"{source}:{min_p}-{max_p}"
        if rkey in used_ranges:
            continue
        used_ranges.add(rkey)
        parts = [chunk_store.texts[wi] for wi in sorted(window)]
        full_text = "\n\n".join(parts)
        boost_final = _keyword_boost(full_text.lower(), q_lower, terms)
        final_score = emb_sim * (1.0 + boost_final) + sent_score * 0.3
        merged.append((full_text, final_score, source, min_p, max_p, boost_final))
        if len(merged) >= top_k:
            break

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def _get_chunk_meta(chunk_store, idx):
    """Метаданные чанка: (source_file, page_num).

    Формат метаданных менялся (S-12): раньше это был dict, теперь — список
    dict'ов, по одному на каждый источник документа. Поддерживаются оба
    формата (и JSON-строка), иначе m.get() бросает AttributeError, исключение
    глушится и функция возвращает ("", 0) для КАЖДОГО чанка — из-за чего все
    вызовы с "if not source: continue" отбрасывали кандидатов и 8 режимов
    меню возвращали пустоту (P0-2).
    """
    try:
        row = chunk_store.get_row(chunk_store.hash_ids[idx])
        m = row.get("metadata", {}) if isinstance(row, dict) else {}
        if isinstance(m, str):
            m = json.loads(m)
        if isinstance(m, list):
            m = m[0] if m else {}  # список источников → первичный источник
        if not isinstance(m, dict):
            return "", 0
        return str(m.get("source_file", "")), int(m.get("page_num", 0))
    except Exception:
        return "", 0


def search_bm25_hybrid(hipporag, query, top_k=10, candidate_pool=200):
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()
    q_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)
    candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]
    scored = []
    for idx in candidate_idx:
        if idx >= n_chunks:
            continue
        text = chunk_store.texts[idx]
        tl = text.lower()
        emb = float(emb_scores[idx])
        boost = _keyword_boost(tl, q_lower, terms)
        final = emb * (1.0 + boost)
        scored.append((text, final, emb, float(bm25_scores[idx]), boost))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_50 = scored[:50]
    rest = scored[50:]
    reranked = _rerank_with_crossencoder(query, top_50)
    return (reranked + rest)[:top_k]


def search_bm25_hybrid_ru(hipporag, query, top_k=10, candidate_pool=200):
    """BM25 + RU-CE + Query expansion (оригинальная версия)."""
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()
    expanded_terms = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + expanded_terms))
    bm25_scores = bm25.get_scores(q_tokens)
    candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]
    scored = []
    for idx in candidate_idx:
        if idx >= n_chunks:
            continue
        text = chunk_store.texts[idx]
        tl = text.lower()
        emb = float(emb_scores[idx])
        boost = _keyword_boost(tl, q_lower, terms)
        final = emb * (1.0 + boost)
        scored.append((text, final, emb, float(bm25_scores[idx]), boost))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_50 = scored[:50]
    rest = scored[50:]
    reranked = _rerank_with_crossencoder_ru(query, top_50)
    return (reranked + rest)[:top_k]


def search_rrf_context(hipporag, query, top_k=10, candidate_pool=150, context_window=1):
    """
    Гибрид RRF (BM25 + Embedding) + контекстное окно + RU-CE.
    """
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]

    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_raw = np.dot(all_emb, q_emb.T).squeeze()

    expanded_terms = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + expanded_terms))
    bm25_raw = bm25.get_scores(q_tokens)

    # Только top-candidates от BM25
    candidate_idx = set(np.argsort(bm25_raw)[::-1][:candidate_pool])

    # RRF среди кандидатов + sign dominance + keyword boost
    bm25_rank = {idx: pos for pos, idx in enumerate(np.argsort(bm25_raw)[::-1])}
    emb_rank = {idx: pos for pos, idx in enumerate(np.argsort(emb_raw)[::-1])}
    k = 60

    scored = []
    for idx in candidate_idx:
        rrf = 1 / (k + bm25_rank.get(idx, 1e6)) + 1 / (k + emb_rank.get(idx, 1e6))
        tl = chunk_store.texts[idx].lower()
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        scored.append((idx, rrf, float(emb_raw[idx]), float(bm25_raw[idx]), boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    # CE rerank top 30
    top_30 = [(chunk_store.texts[idx], s, e, b, boost) for idx, s, e, b, boost in scored[:30]]
    try:
        reranked = _rerank_with_crossencoder_ru(query, top_30)
        text_to_idx = {chunk_store.texts[i]: i for i in range(n_chunks)}
        reranked_idx = [text_to_idx[r[0]] for r in reranked if r[0] in text_to_idx]
    except Exception:
        reranked_idx = [scored[i][0] for i in range(min(30, len(scored)))]
    reranked_idx.extend([scored[i][0] for i in range(min(30, len(scored)), len(scored))])

    # Контекстное окно (только ±1 страница)
    used_ranges = set()
    merged = []
    for idx in reranked_idx:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-context_window, context_window + 1):
            ni = idx + offset
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= context_window:
                window.add(ni)
        pages = set()
        for wi in window:
            _, wp = _get_chunk_meta(chunk_store, wi)
            pages.add(wp)
        min_p, max_p = min(pages), max(pages)
        rkey = f"{source}:{min_p}-{max_p}"
        if rkey in used_ranges:
            continue
        used_ranges.add(rkey)
        parts = [chunk_store.texts[wi] for wi in sorted(window)]
        full_text = "\n\n".join(parts)
        tl_full = full_text.lower()
        boost = _keyword_boost(tl_full, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final = float(emb_raw[idx]) * (1.0 + boost)
        merged.append((full_text, final, source, min_p, max_p, boost))
        if len(merged) >= top_k:
            break

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


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


def search_small_to_big(hipporag, query, top_k=10, candidate_pool=100, context_window=2):
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()
    q_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)
    candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]
    scored = []
    for idx in candidate_idx:
        if idx >= n_chunks:
            continue
        text = chunk_store.texts[idx]
        tl = text.lower()
        emb = float(emb_scores[idx])
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final = emb * (1.0 + boost)
        scored.append((idx, final, emb, float(bm25_scores[idx]), boost))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Cross-encoder rerank top 50
    try:
        ce = _get_crossencoder()
        top_50 = scored[:50]
        pairs = [(query, chunk_store.texts[idx][:512]) for idx, *_ in top_50]
        xc_scores = ce.predict(pairs, show_progress_bar=False)
        for i, (idx, final, emb, bm25_s, boost) in enumerate(top_50):
            xc = float(xc_scores[i])
            final = emb * 0.8 * (1.0 + boost) + xc * 0.2
            scored[i] = (idx, final, emb, bm25_s, boost, xc)
    except Exception:
        pass

    scored.sort(key=lambda x: x[1], reverse=True)
    used_ranges = set()
    merged = []
    context_window = 1

    for idx, final, emb, bm25_s, boost, *_ in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-context_window * 2, context_window * 2 + 1):
            ni = idx + offset
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= context_window:
                window.add(ni)
        pages = set()
        for wi in window:
            _, wp = _get_chunk_meta(chunk_store, wi)
            pages.add(wp)
        min_p, max_p = min(pages), max(pages)
        rkey = f"{source}:{min_p}-{max_p}"
        if rkey in used_ranges:
            continue
        used_ranges.add(rkey)
        parts = [chunk_store.texts[wi] for wi in sorted(window)]
        full_text = "\n\n".join(parts)
        boost_final = _keyword_boost(full_text.lower(), q_lower, terms)
        final_score = final * (1.0 + boost_final * 0.5)
        merged.append((full_text, final_score, source, min_p, max_p, boost_final))
        if len(merged) >= top_k:
            break

    # Обрезаем начало до заголовка нужного знака зодиака
    sign_in_result = _sign_in_query(q_lower)
    if sign_in_result:
        import re as _re

        trimmed = []
        for text, score, source, min_p, max_p, boost in merged:
            for form in ZODIAC_FORMS.get(sign_in_result, [sign_in_result]):
                pattern = rf"(?:^|[.!\n])\s*{_re.escape(form.upper())}(?:\s|[:,.!])"
                m = _re.search(pattern, text, _re.IGNORECASE)
                if m:
                    cut_pos = m.start()
                    if cut_pos > 0:
                        text = text[cut_pos:]
                    text = text.lstrip(" .\n\t,;:!?")
                    break
            trimmed.append((text, score, source, min_p, max_p, boost))
        merged = trimmed

    # Дедупликация: убираем дубликаты по содержимому
    seen_texts = set()
    deduped = []
    for text, score, source, min_p, max_p, boost in merged:
        # Сравниваем по первым 200 символам
        key = text[:200].lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped.append((text, score, source, min_p, max_p, boost))
    merged = deduped

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


_COLBERT = None
_COLBERT_LINEAR = None
_COLBERT_BERT = None


def _get_colbert():
    """Загрузить ColBERT-v2 (лениво): BERT + linear projection (128-dim)."""
    global _COLBERT, _COLBERT_LINEAR, _COLBERT_BERT
    if _COLBERT is None:
        import os as _os

        from safetensors.torch import load_file
        from sentence_transformers import SentenceTransformer

        print("  ⏳ Загрузка ColBERT-v2...", flush=True)
        st = SentenceTransformer("colbert-ir/colbertv2.0")
        _COLBERT_BERT = st[0].auto_model
        _COLBERT_BERT.eval()
        hub = _os.path.expanduser(
            "~/.cache/huggingface/hub/models--colbert-ir--colbertv2.0/snapshots"
        )
        if _os.path.exists(hub):
            snaps = _os.listdir(hub)
            if snaps:
                safetensors = load_file(_os.path.join(hub, snaps[0], "model.safetensors"))
                _COLBERT_LINEAR = safetensors["linear.weight"]  # (128, 768)
        _COLBERT = st
        print("  ✅ ColBERT-v2 загружен", flush=True)
    return _COLBERT


def _colbert_encode(text, max_length=512):
    """Encode text with ColBERT → 128-dim token embeddings + mask."""
    import torch
    import torch.nn.functional as F

    tokens = _COLBERT.tokenize([text])
    for k, v in tokens.items():
        if isinstance(v, torch.Tensor) and v.shape[1] > max_length:
            tokens[k] = v[:, :max_length]
    # Токенизатор возвращает тензоры на CPU, а модель может быть на CUDA:
    # token_type_ids в слое эмбеддингов идёт через index_select и падает с
    # "Expected all tensors to be on the same device" (P2-7).
    device = next(_COLBERT_BERT.parameters()).device
    tokens = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in tokens.items()}
    with torch.no_grad():
        out = _COLBERT_BERT(**tokens)
        emb = out.last_hidden_state
        if _COLBERT_LINEAR is not None:
            emb = torch.matmul(emb, _COLBERT_LINEAR.T.to(emb.device))
        emb = F.normalize(emb, dim=-1)
    return emb, tokens["attention_mask"]


def _colbert_score(q_emb, q_mask, d_emb, d_mask):
    """MaxSim late interaction score."""
    import torch

    sim = torch.matmul(q_emb, d_emb.transpose(1, 2))
    qm = q_mask.unsqueeze(-1).float()
    dm = d_mask.unsqueeze(1).float()
    sim = sim * qm * dm
    max_sim = sim.max(dim=-1).values
    score = (max_sim * qm.squeeze(-1)).sum(dim=-1)
    score = score / qm.squeeze(-1).sum(dim=-1).clamp(min=1)
    return score.item()


def search_colbert(hipporag, query, top_k=10, candidate_pool=200):
    """
    Поиск: RRF (BM25 + эмбеддинги) → ColBERT-v2 late interaction reranking.
    Токен-уровневый поиск: каждый токен запроса сравнивается с каждым токеном чанка.
    Медленнее, но точнее для сложных cross-chunk запросов.
    """
    import time as _time

    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    exp = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + exp))
    bm25_scores = bm25.get_scores(q_tokens)

    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()

    bm25_r = np.argsort(bm25_scores)[::-1]
    emb_r = np.argsort(emb_scores)[::-1]
    bm25_p = {idx: p for p, idx in enumerate(bm25_r)}
    emb_p = {idx: p for p, idx in enumerate(emb_r)}
    k = 60
    rrf = [1 / (k + bm25_p.get(i, 1e6)) + 1 / (k + emb_p.get(i, 1e6)) for i in range(n_chunks)]
    candidates = np.argsort(rrf)[::-1][:candidate_pool]
    if len(candidates) == 0:
        return []

    _get_colbert()
    print("  🧠 ColBERT late interaction (30 candidates)...", flush=True)
    t0 = _time.time()
    q_ce, q_cm = _colbert_encode(query)

    top30 = candidates[:30]
    rest = candidates[30:]

    scored = []
    for pos, idx in enumerate(top30):
        t1 = _time.time()
        d_ce, d_cm = _colbert_encode(chunk_store.texts[idx][:512])
        col_score = _colbert_score(q_ce, q_cm, d_ce, d_cm)
        dt = _time.time() - t1
        tl = chunk_store.texts[idx].lower()
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        # ColBERT (70%) + embedding (30%), boost применяется к ColBERT
        final = col_score * 0.7 * (1.0 + boost * 0.5) + float(emb_scores[idx]) * 0.3
        scored.append((idx, final, float(emb_scores[idx]), col_score, boost))
        if pos < 3 or pos == len(top30) - 1:
            print(f"     [{pos+1}/30] score={col_score:.4f} ({dt:.1f}s)", flush=True)

    for idx in rest:
        tl = chunk_store.texts[idx].lower()
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final = float(emb_scores[idx]) * (1.0 + boost)
        scored.append((idx, final, float(emb_scores[idx]), 0, boost))

    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"  ✅ ColBERT: {_time.time()-t0:.0f} сек total", flush=True)

    # Контекстное окно ±1
    used = set()
    merged = []
    for idx, final, emb_sim, col_s, boost in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        w = set([idx])
        for off in range(-1, 2):
            ni = idx + off
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= 1:
                w.add(ni)
        pages = set()
        for wi in w:
            _, wp = _get_chunk_meta(chunk_store, wi)
            pages.add(wp)
        mp, xp = min(pages), max(pages)
        rk = f"{source}:{mp}-{xp}"
        if rk in used:
            continue
        used.add(rk)
        parts = [chunk_store.texts[wi] for wi in sorted(w)]
        ft = "\n\n".join(parts)
        bf = _keyword_boost(ft.lower(), q_lower, terms)
        fs = emb_sim * (1.0 + bf) + max(col_s, 0) * 0.3
        merged.append((ft, fs, source, mp, xp, bf))
        if len(merged) >= top_k:
            break

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def search_small_to_big_ru(hipporag, query, top_k=10, candidate_pool=300, context_window=1):
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()
    # Query expansion для BM25
    expanded_terms = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + expanded_terms))
    bm25_scores = bm25.get_scores(q_tokens)
    candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]
    scored = []
    for idx in candidate_idx:
        if idx >= n_chunks:
            continue
        text = chunk_store.texts[idx]
        tl = text.lower()
        emb = float(emb_scores[idx])
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final = emb * (1.0 + boost)
        scored.append((idx, final, emb, float(bm25_scores[idx]), boost))
    scored.sort(key=lambda x: x[1], reverse=True)

    try:
        ce = _get_crossencoder_ru()
        top_30 = scored[:30]
        pairs = [(query, chunk_store.texts[idx][:512]) for idx, *_ in top_30]
        xc_scores = ce.predict(pairs, show_progress_bar=False)
        for i, (idx, final, emb, bm25_s, boost) in enumerate(top_30):
            xc = float(xc_scores[i])
            final = emb * 0.8 * (1.0 + boost) + xc * 0.2
            scored[i] = (idx, final, emb, bm25_s, boost, xc)
    except Exception:
        pass

    scored.sort(key=lambda x: x[1], reverse=True)
    used_ranges = set()
    merged = []

    for idx, final, emb, bm25_s, boost, *_ in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-context_window * 2, context_window * 2 + 1):
            ni = idx + offset
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= context_window:
                window.add(ni)
        pages = set()
        for wi in window:
            _, wp = _get_chunk_meta(chunk_store, wi)
            pages.add(wp)
        min_p, max_p = min(pages), max(pages)
        rkey = f"{source}:{min_p}-{max_p}"
        if rkey in used_ranges:
            continue
        used_ranges.add(rkey)
        parts = [chunk_store.texts[wi] for wi in sorted(window)]
        full_text = "\n\n".join(parts)
        boost_final = _keyword_boost(full_text.lower(), q_lower, terms)
        final_score = final * (1.0 + boost_final * 0.5)
        merged.append((full_text, final_score, source, min_p, max_p, boost_final))
        if len(merged) >= top_k:
            break

    sign_in_result = _sign_in_query(q_lower)
    if sign_in_result:
        import re as _re

        trimmed = []
        for text, score, source, min_p, max_p, boost in merged:
            for form in ZODIAC_FORMS.get(sign_in_result, [sign_in_result]):
                pattern = rf"(?:^|[.!\n])\s*{_re.escape(form.upper())}(?:\s|[:,.!])"
                m = _re.search(pattern, text, _re.IGNORECASE)
                if m:
                    cut_pos = m.start()
                    if cut_pos > 0:
                        text = text[cut_pos:]
                    text = text.lstrip(" .\n\t,;:!?")
                    break
            trimmed.append((text, score, source, min_p, max_p, boost))
        merged = trimmed

    seen_texts = set()
    deduped = []
    for text, score, source, min_p, max_p, boost in merged:
        key = text[:200].lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped.append((text, score, source, min_p, max_p, boost))
    merged = deduped

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def _filter_by_section(candidate_idx, chunk_store, q_lower, keep_general=0.15):
    """
    Отфильтровать кандидаты: оставить только чанки из раздела
    релевантного знака зодиака + немного общих чанков для контекста.
    """
    sign = _sign_in_query(q_lower)
    if not sign:
        return candidate_idx

    filtered = []
    general = []
    for idx in candidate_idx:
        tl = chunk_store.texts[idx].lower()
        if _count_sign(tl, sign) > 0:
            filtered.append(idx)
        else:
            general.append(idx)

    keep_gen = max(int(len(candidate_idx) * keep_general), 5)
    result = filtered + general[:keep_gen]
    if not filtered:
        return candidate_idx  # если не нашли ничего — возвращаем всё
    return result


def _llm_filter_results(query, results, max_results=5):
    """
    Универсальный LLM-фильтр для результатов любого метода поиска.
    Отправляет топ-10 результатов в LLM, просит отобрать релевантные.
    """
    import json as _json

    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
    try:
        r = httpx.get(f"{api_url}/v1/models", timeout=3)
        if r.status_code != 200:
            return results[:max_results]
    except:
        return results[:max_results]

    if not results:
        return results

    # Собираем топ-10 результатов в json для LLM
    candidates = []
    for i, r in enumerate(results[:10]):
        text = r[0][:300] if isinstance(r, (list, tuple)) and len(r) > 0 else str(r)[:300]
        source = r[2] if isinstance(r, (list, tuple)) and len(r) > 2 else f"result_{i}"
        candidates.append({"id": i, "source": str(source), "text": text})

    prompt = {
        "model": LLM_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты — фильтр качества поиска. Оцени каждый результат по критериям:\n"
                    "- Содержит ли текст ответ на вопрос?\n"
                    "- Есть ли в тексте ключевые слова из вопроса?\n\n"
                    "Верни JSON-список id релевантных результатов. "
                    'Формат: {"relevant_ids": [0, 2, 5]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {query}\n\n"
                    f"Результаты: {_json.dumps(candidates, ensure_ascii=False)}\n\n"
                    f"Какие id релевантны?"
                ),
            },
        ],
        "max_tokens": 200,
        "temperature": 0.1,
    }
    try:
        resp = httpx.post(f"{api_url}/v1/chat/completions", json=prompt, timeout=300)
        if resp.status_code == 200:
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content and msg.get("reasoning_content"):
                rc = msg["reasoning_content"]
                lines = [l.strip() for l in rc.split("\n") if l.strip()]
                content = lines[-1].strip("\"'").replace("**", "") if lines else content
            # Извлекаем JSON
            import re as _re

            json_match = _re.search(r'\{.*"relevant_ids".*\}', content, _re.DOTALL)
            if json_match:
                filtered = _json.loads(json_match.group())
                relevant = filtered.get("relevant_ids", [])
                return [results[i] for i in relevant if i < len(results)]
    except:
        pass
    return results[:max_results]


def search_with_llm_filter(hipporag, query):
    """
    Комбинация: поиск через Small-to-big + RU-CE (метод 6)
    + LLM фильтрация нерелевантных результатов.
    """
    results = search_small_to_big_ru(hipporag, query, top_k=10)
    if not results:
        return []
    print(f"  🤖 LLM фильтрация {len(results)} результатов...", flush=True)
    filtered = _llm_filter_results(query, results, max_results=5)
    print(f"  ✅ После фильтра: {len(filtered)} результатов", flush=True)
    return filtered if filtered else results[:3]


_QWEN_MODEL = None
_FRIDA_MODEL = None


def _get_qwen():
    global _QWEN_MODEL
    if _QWEN_MODEL is None:
        from sentence_transformers import SentenceTransformer

        print("  ⏳ Загрузка Qwen3-Embedding-0.6B...", flush=True)
        _QWEN_MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        print("  ✅ Qwen3 загружен", flush=True)
    return _QWEN_MODEL


def _get_frida():
    global _FRIDA_MODEL
    if _FRIDA_MODEL is None:
        from sentence_transformers import SentenceTransformer

        print("  ⏳ Загрузка FRIDA (0.8B)...", flush=True)
        _FRIDA_MODEL = SentenceTransformer("ai-forever/FRIDA")
        print("  ✅ FRIDA загружена", flush=True)
    return _FRIDA_MODEL


def search_qwen_frida(hipporag, query, top_k=10):
    """
    Метод 18: BM25 + Qwen3-Embedding (dense) + FRIDA (русский reranker) + LLM фильтр.
    """
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    exp = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + exp))
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_candidates = np.argsort(bm25_scores)[::-1][:300]

    # Qwen3 dense embedding (только для топ-50 кандидатов)
    print("  🤖 Qwen3-Embedding...", flush=True)
    qwen = _get_qwen()
    q_emb = qwen.encode([f"search_query: {query}"])[0]
    top50 = bm25_candidates[:50]
    top_texts = [chunk_store.texts[i] for i in top50]
    top_embs = qwen.encode([f"search_document: {t}" for t in top_texts])

    # RRF: BM25 + Qwen3
    bm25_pos = {idx: p for p, idx in enumerate(bm25_candidates)}
    qwen_scores = np.dot(top_embs, q_emb)
    qwen_rank = np.argsort(qwen_scores)[::-1]
    qwen_pos = {top50[i]: p for p, i in enumerate(qwen_rank)}
    k = 60
    scored = []
    for idx in top50:
        rrf = 1 / (k + bm25_pos.get(idx, 1e6)) + 1 / (k + qwen_pos.get(idx, 1e6))
        boost = _keyword_boost(chunk_store.texts[idx].lower(), q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        scored.append((idx, rrf, float(qwen_scores[top50.tolist().index(idx)]), boost))
    for idx in bm25_candidates[50:]:
        scored.append((idx, 1 / (k + bm25_pos.get(idx, 1e6)), 0, 0))
    scored.sort(key=lambda x: x[1], reverse=True)

    # FRIDA reranking топ-20
    print("  🤖 FRIDA reranking...", flush=True)
    frida = _get_frida()
    top20 = scored[:20]
    frida_scores = frida.encode([f"search_query: {query}" for _ in top20])
    frida_docs = frida.encode(
        [f"search_document: {chunk_store.texts[idx][:512]}" for idx, *_ in top20]
    )
    frida_sims = (frida_scores * frida_docs).sum(axis=1)

    for i, (idx, rrf, qs, boost) in enumerate(top20):
        fs = float(frida_sims[i])
        final = fs * 0.5 + rrf * 0.3 + qs * 0.2 * (1.0 + boost)
        top20[i] = (idx, final, fs, boost)
    top20.sort(key=lambda x: x[1], reverse=True)
    all_results = [(chunk_store.texts[idx], float(fs), idx) for idx, final, fs, boost in top20]
    all_results += [(chunk_store.texts[idx], 0, idx) for idx, rrf, qs, boost in scored[20:]]

    # LLM фильтрация
    print("  🤖 LLM фильтрация...", flush=True)
    filtered = _llm_filter_results(query, all_results, max_results=5)

    # Контекстное окно ±1 для отфильтрованных
    used_ranges = set()
    merged = []
    for text, fs, idx in filtered:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for off in range(-1, 2):
            ni = idx + off
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= 1:
                window.add(ni)
        pages = set()
        for wi in window:
            _, wp = _get_chunk_meta(chunk_store, wi)
            pages.add(wp)
        mp, xp = min(pages), max(pages)
        rk = f"{source}:{mp}-{xp}"
        if rk in used_ranges:
            continue
        used_ranges.add(rk)
        parts = [chunk_store.texts[wi] for wi in sorted(window)]
        ft = "\n\n".join(parts)
        bf = _keyword_boost(ft.lower(), q_lower, terms)
        merged.append((ft, fs, source, mp, xp, bf))
        if len(merged) >= top_k:
            break

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def search_graphrag_agent(hipporag, query):
    """
    Метод 19: GraphRAG Agent.
    Router → Search → Self-Correction → LLM Answer.
    Анализирует запрос, выбирает лучший метод, проверяет качество,
    при необходимости переформулирует и пробует снова.
    """
    import time as _time

    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
    terms, q_lower = split_query_terms(query)
    exp_terms = _expand_query(q_lower, terms)

    def has_any_term(text, term_list):
        tl = text.lower()
        return any(t in tl for t in term_list if len(t) > 2)

    # --- Шаг 1: Router — определяем тип запроса ---
    is_global = any(w in q_lower for w in ["перечисли", "список", "все", "какие", "сколько"])
    is_relation = any(w in q_lower for w in ["связан", "отличается", "общего", "связь"])
    is_multi_hop = len(terms) >= 4 and not is_global

    # --- Шаг 2: Выбираем метод ---
    s2b = lambda q: search_small_to_big_ru(hipporag, q, top_k=8)
    bm25ru = lambda q: search_bm25_hybrid_ru(hipporag, q, top_k=8)
    rrf = lambda q: search_rrf_context(hipporag, q, top_k=8)
    filt = lambda q: search_filtered(hipporag, q, top_k=8)
    srag = lambda q: search_self_rag(hipporag, q)

    methods = []
    if is_global:
        methods = [(rrf, "RRF"), (s2b, "Small→Big")]
    elif is_relation:
        methods = [(filt, "Filter"), (rrf, "RRF")]
    elif is_multi_hop:
        methods = [(srag, "Self-RAG"), (s2b, "Small→Big")]
    else:
        methods = [(bm25ru, "BM25+RU"), (s2b, "Small→Big")]

    # Исправляем search_and_check: передаём lambda без top_k
    def search_and_check2(fn, name, tk=8):
        t0 = _time.time()
        try:
            res = fn(query)
        except:
            res = []
        dt = _time.time() - t0
        if not res:
            return [], 0, dt
        relevant = sum(
            1
            for r in res
            if has_any_term(r[0] if isinstance(r, (list, tuple)) else str(r), exp_terms)
        )
        return res, relevant / max(len(res), 1), dt

    # --- Шаг 3: Каскадный поиск с самокоррекцией ---
    best_results = []
    best_quality = 0
    best_method = ""
    log = []

    for fn, name in methods:
        if best_quality >= 0.8:
            break
        res, quality, dt = search_and_check2(fn, name)
        log.append(f"{name}: quality={quality:.0%} ({dt:.0f}s)")
        print(f"  📊 {log[-1]}", flush=True)
        if res and (not best_results or quality > best_quality):
            best_results = res
            best_quality = quality
            best_method = name

    # Если все методы дали плохой результат — пробуем LLM переформулировку
    if best_quality < 0.3:
        try:
            r = httpx.get(f"{api_url}/v1/models", timeout=3)
            if r.status_code == 200:
                prompt = {
                    "model": LLM_NAME,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Переформулируй запрос для поиска. Выдели 2-5 ключевых слов.",
                        },
                        {"role": "user", "content": f"Запрос: {query}"},
                    ],
                    "max_tokens": 100,
                    "temperature": 0.1,
                }
                resp = httpx.post(f"{api_url}/v1/chat/completions", json=prompt, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    new_q = (msg.get("content") or "").strip().strip("\"'")

                    if new_q and 5 < len(new_q) < 100:
                        print(f"  🔄 Переформулировка: '{query}' → '{new_q}'", flush=True)
                        rrf_retry = lambda q: search_rrf_context(hipporag, q, top_k=10)
                        res, quality, dt = search_and_check2(rrf_retry, "RRF(retry)", tk=10)
                        if quality > best_quality:
                            best_results = res
                            best_method = "RRF(retry)"
                            log.append(f"RRF(retry): quality={quality:.0%}")
        except:
            pass

    # Если лучший метод — Self-RAG, используем его ответ напрямую
    if best_method == "Self-RAG" and best_results:
        return best_results[:1]

    # --- Шаг 4: LLM формирует ответ ---
    answer = None
    if best_results:
        # Собираем текст
        chunks_text = ""
        for i, r in enumerate(best_results[:5]):
            doc = r[0] if isinstance(r, (list, tuple)) else str(r)
            src = r[2] if isinstance(r, (list, tuple)) and len(r) > 2 else ""
            src_str = str(src) if src else ""
            chunks_text += f"\n[Источник {i+1} {src_str}]\n{doc[:1000]}\n"

        answer = None
        try:
            r = httpx.get(f"{api_url}/v1/models", timeout=3)
            if r.status_code == 200:
                prompt = {
                    "model": LLM_NAME,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Ответь на вопрос на основе источников.\n\n"
                                "АЛГОРИТМ:\n"
                                "1. Найди в источниках прямые ответы и соответствия\n"
                                "2. Для вопросов вида «какая планета отвечает за зону» — "
                                "ищи «знаку соответствует зона» + «управитель знака — планета»\n"
                                "3. Для вопросов «перечисли всё» — выведи полный список\n"
                                "4. Для остальных вопросов (автор, книга, понятие) — приведи прямое "
                                "содержание найденного, дословно как в источниках\n"
                                "5. Если нужная информация есть, но сформулирована иначе, чем "
                                "вопрос — приведи её. Отсутствием информации считай только "
                                "полное отсутствие чего-либо по теме.\n"
                                "6. Ссылайся на источники: [Источник 1]\n"
                                "7. «нет данных» пиши ТОЛЬКО если в источниках действительно нет "
                                "ничего по теме вопроса"
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Вопрос: {query}\n\nИсточники:\n{chunks_text}",
                        },
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.15,
                }
                resp = httpx.post(f"{api_url}/v1/chat/completions", json=prompt, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    answer = (msg.get("content") or "").strip()
                    if not answer and msg.get("reasoning_content"):
                        rc = msg["reasoning_content"]
                        lines = [l.strip() for l in rc.split("\n") if l.strip()]
                        answer = lines[-1].strip("\"'").replace("**", "") if lines else answer
        except:
            pass

    # Раньше функция заканчивалась здесь без return: маршрутизация, каскадный
    # поиск и ответ LLM выполнялись, а результат выбрасывался — режим 19 всегда
    # возвращал None и показывал «Ничего не найдено».
    if answer:
        return [(answer, 999.0, 0, 0, 0)]
    return best_results[:1]


# Библиографические вопросы: ответ — название книги/автор, а не отдельное
# предложение корпуса. Sentence-level поиск их структурно не находит (P1-7).
BIBLIO_QUERY_PATTERNS = (
    r"как(?:ие|ую|их|ой|ого)\s+книг",
    r"названи[еяй]\s+книг",
    r"кто\s+(?:автор|написал|писатель)",
    r"что\s+написал",
    r"как(?:ие|ой|ую|их)\s+(?:произведени|работ)",
    r"список\s+(?:книг|произведени)",
    r"библиограф",
    r"издани[еяй]\s+книг",
    r"какая\s+книга",
)


def _is_bibliographic_query(query):
    """Вопрос про книги/автора/издания, а не про содержание (P1-7)."""
    ql = query.lower()
    return any(re.search(pattern, ql) for pattern in BIBLIO_QUERY_PATTERNS)


def search_sentence_level(hipporag, query, top_k=10):
    if _is_bibliographic_query(query):
        # Название книги — не предложение: «Психологический фэншуй: Как
        # прокачать «плохой» фэншуй» не попадает в sentence-level выдачу.
        # Для библиографических вопросов уходим на chunk-level поиск (P1-7).
        print("  Библиографический вопрос → chunk-level (BM25+RU-CE)", flush=True)
        return search_bm25_hybrid_ru(hipporag, query, top_k=top_k)
    import re as _re

    chunk_store = hipporag.chunk_embedding_store
    terms, q_lower = split_query_terms(query)
    exp = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + exp))
    if not hasattr(search_sentence_level, "sentences_cache"):
        sents = []
        for idx, text in enumerate(chunk_store.texts):
            for s in _re.split(r"(?<=[.!?])\s+", text):
                s = s.strip()
                if len(s) > 15:
                    sents.append((s.lower(), s, idx))
        search_sentence_level.sentences_cache = sents
        print(f"  Предложений: {len(sents)}", flush=True)
    sentences = search_sentence_level.sentences_cache
    sent_tokens = [_tokenize(s[0]) for s in sentences]
    sent_bm25 = BM25Okapi(sent_tokens)
    sent_scores = sent_bm25.get_scores(q_tokens)
    sent_rank = np.argsort(sent_scores)[::-1]
    seen_chunks = set()
    merged = []
    for si in sent_rank:
        if sent_scores[si] == 0:
            continue
        sent_lower, sent_orig, chunk_idx = sentences[si]
        if chunk_idx in seen_chunks:
            continue
        seen_chunks.add(chunk_idx)
        source, page = _get_chunk_meta(chunk_store, chunk_idx)
        full_text = chunk_store.texts[chunk_idx]
        boost = _keyword_boost(full_text.lower(), q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, chunk_idx, q_lower)
        boost += sign_score
        final_score = float(sent_scores[si]) * (1.0 + boost)
        display_text = (
            full_text.replace(sent_orig, f"\u2192{sent_orig}\u2190", 1)
            if sent_orig in full_text
            else full_text
        )
        merged.append((display_text, final_score, source or "", page, 0, boost))
        if len(merged) >= top_k:
            break
    if not merged:
        return search_small_to_big_ru(hipporag, query, top_k=min(5, top_k))
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def _dat_evaluate(query, sent_top1, dense_top1):
    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
    if not sent_top1 or not dense_top1:
        return 3, 3
    prompt_text = (
        "Ты — оценщик качества поиска. Оцени два результата поиска по шкале от 0 до 5.\n\n"
        "Критерии:\n"
        "5 — документ напрямую отвечает на вопрос\n"
        "3-4 — концептуально близок, содержит частичный ответ или связанные сущности\n"
        "1-2 — слабо связан, вводит в заблуждение\n"
        "0 — полностью нерелевантен\n\n"
        f"Вопрос: {query}\n\n"
        f"Результат 1 (поиск по предложениям):\n{sent_top1[:500]}\n\n"
        f"Результат 2 (семантический поиск):\n{dense_top1[:500]}\n\n"
        "Верни два числа через пробел: оценка_1 оценка_2.\nПример: 4 3"
    )
    for attempt in range(2):
        try:
            r = httpx.post(
                f"{api_url}/v1/chat/completions",
                json={
                    "model": LLM_NAME,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": 50,
                    "temperature": 0.1,
                },
                timeout=300,
            )
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"].get("content", "").strip()
                nums = re.findall(r"\d+", content)
                if len(nums) >= 2:
                    return min(int(nums[0]), 5), min(int(nums[1]), 5)
                if attempt == 0:
                    prompt_text += "\n\nОтветь ТОЛЬКО двумя числами через пробел."
                    continue
        except:
            if attempt == 0:
                continue
    return 3, 3


def _dat_compute_alpha(score_sentence, score_dense):
    if score_sentence == 0 and score_dense == 0:
        return 0.5
    if score_dense == 5 and score_sentence != 5:
        return 1.0
    if score_sentence == 5 and score_dense != 5:
        return 0.0
    denom = score_dense + score_sentence
    if denom == 0:
        return 0.5
    return round(score_dense / denom, 1)


def _dat_combine_results(sentence_results, dense_results, alpha, top_k=15):
    combined = {}
    s_scores = [r[1] for r in sentence_results] if sentence_results else [0]
    s_max, s_min = max(s_scores), min(s_scores)
    s_range = s_max - s_min if s_max > s_min else 1
    d_scores = [r[1] for r in dense_results] if dense_results else [0]
    d_max, d_min = max(d_scores), min(d_scores)
    d_range = d_max - d_min if d_max > d_min else 1
    for r in sentence_results:
        text, score = r[0], r[1]
        src = r[2] if len(r) > 2 else ""
        pg = r[3] if len(r) > 3 else 0
        norm = (score - s_min) / s_range
        if str(text)[:200].strip() in combined:
            combined[str(text)[:200].strip()]["s_norm"] = norm
        else:
            combined[str(text)[:200].strip()] = {
                "text": text,
                "source": src,
                "page": pg,
                "s_norm": norm,
                "d_norm": 0.0,
            }
    for r in dense_results:
        text, score = str(r[0]) if not isinstance(r[0], str) else r[0], r[1]
        src = r[2] if len(r) > 2 else ""
        pg = r[3] if len(r) > 3 else 0
        norm = (score - d_min) / d_range
        k = str(text)[:200].strip()
        if k in combined:
            combined[k]["d_norm"] = norm
        else:
            combined[k] = {"text": text, "source": src, "page": pg, "s_norm": 0.0, "d_norm": norm}
    for v in combined.values():
        v["final"] = alpha * v["d_norm"] + (1 - alpha) * v["s_norm"]
    sorted_items = sorted(combined.values(), key=lambda x: x["final"], reverse=True)
    return [
        (v["text"], v["final"], v["source"], v["page"], 0, v["final"]) for v in sorted_items[:top_k]
    ]


def search_dat_sentence_llm(hipporag, query, top_k=10):
    import time as _time

    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
    t0 = _time.time()
    print("  DAT Sentence-level + LLM", flush=True)
    sentence_results = search_sentence_level(hipporag, query, top_k=30)
    t1 = _time.time()
    print(f"  Sentence-level: {len(sentence_results)} рез. ({t1-t0:.1f}с)", flush=True)
    dense_results = search_small_to_big_ru(hipporag, query, top_k=30)
    t2 = _time.time()
    print(f"  Dense: {len(dense_results)} рез. ({t2-t1:.1f}с)", flush=True)
    sent_top1_text = (
        sentence_results[0][0][:500] if sentence_results and len(sentence_results[0]) > 0 else ""
    )
    dense_top1_text = (
        dense_results[0][0][:500] if dense_results and len(dense_results[0]) > 0 else ""
    )
    s_score, d_score = _dat_evaluate(query, sent_top1_text, dense_top1_text)
    alpha = _dat_compute_alpha(s_score, d_score)
    t3 = _time.time()
    print(
        f"  DAT: sentence={s_score}/5, dense={d_score}/5 -> alpha={alpha} ({t3-t2:.1f}с)",
        flush=True,
    )
    combined = _dat_combine_results(sentence_results, dense_results, alpha, top_k=15)
    if not combined:
        combined = sentence_results[:top_k]
    t4 = _time.time()
    print(f"  Combined: {len(combined)} рез. ({t4-t3:.1f}с)", flush=True)
    chunks_text = ""
    for i, r in enumerate(combined[:8], 1):
        text = r[0] if isinstance(r, tuple) and len(r) > 0 else str(r)
        src = r[2] if isinstance(r, tuple) and len(r) > 2 else ""
        pg = r[3] if isinstance(r, tuple) and len(r) > 3 else 0
        chunks_text += f"\n[Источник {i} - {src} стр.{pg}]\n{text[:1500]}\n"
    answer = None
    prompt_base = {
        "model": LLM_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты \u2014 эксперт по астрологии и фэншуй. "
                    "Ответь на вопрос, используя ТОЛЬКО текст источников.\n\n"
                    "В тексте знаки зодиака сопоставляются с зонами квартиры через тире: "
                    "\u00abЗНАК \u2014 ЗОНА\u00bb. Например: \u00abОвен \u2014 прихожая, ... \u2014 плита\u00bb.\n\n"
                    "Таблица планет-управителей:\n"
                    "Овен \u2192 Марс, Телец \u2192 Венера, Близнецы \u2192 Меркурий, Рак \u2192 Луна, "
                    "Лев \u2192 Солнце, Дева \u2192 Меркурий, Весы \u2192 Венера, "
                    "Скорпион \u2192 Плутон/Марс, Стрелец \u2192 Юпитер, "
                    "Козерог \u2192 Сатурн, Водолей \u2192 Уран, Рыбы \u2192 Нептун\n\n"
                    "Алгоритм:\n"
                    "1. Найди \u00abЗНАК \u2014 ... ЗОНА\u00bb в источниках\n"
                    "2. Определи планету-управителя знака\n"
                    "3. Вывод: планета отвечает за зону\n"
                    "4. Пример: \u00abОВЕН \u2014 плита\u00bb + Овен\u2192Марс \u2192 Марс отвечает за плиту\n"
                    "5. Алгоритм знак→зона применяй ТОЛЬКО если вопрос про знак или зону. "
                    "На остальные вопросы (книги, автор, издание, понятие) отвечай по "
                    "источникам дословно — например, перечисли названия книг, если они там есть\n"
                    "6. Если нужной информации в источниках действительно нет — напиши "
                    "\u00abнет данных\u00bb"
                ),
            },
            {"role": "user", "content": f"Вопрос: {query}\n\nИсточники:\n{chunks_text}"},
        ],
        "max_tokens": 2048,
        "temperature": 0.15,
    }
    prompts_to_try = [
        prompt_base,
        {
            "model": LLM_NAME,
            "messages": [
                {"role": "user", "content": f"Вопрос: {query}\n\nИсточники:\n{chunks_text}"}
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        },
        {
            "model": LLM_NAME,
            "messages": [{"role": "user", "content": f"{query}\n\n{chunks_text[:500]}"}],
            "max_tokens": 512,
            "temperature": 0.1,
        },
    ]
    for attempt, prompt in enumerate(prompts_to_try):
        try:
            resp = httpx.post(f"{api_url}/v1/chat/completions", json=prompt, timeout=300)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                finish = data["choices"][0].get("finish_reason", "")
                content = (msg.get("content") or "").strip()
                if content:
                    answer = content
                    break
                rc = msg.get("reasoning_content", "") or msg.get("reasoning", "") or ""
                if rc:
                    m = re.search(r"(?:Ответ|Answer)\s*:?\s*(.+)", rc, re.IGNORECASE | re.DOTALL)
                    if m:
                        answer = m.group(1).strip()
                        break
                    m = re.search(r"So the answer is:\s*(.+)", rc, re.IGNORECASE)
                    if m:
                        answer = m.group(1).strip()
                        break
                    lines = [l.strip() for l in rc.split("\n") if l.strip()]
                    if lines:
                        answer = lines[-1].strip("\"'").replace("**", "").strip()
                        break
                print(f"  LLM пусто: попытка {attempt+1}, finish={finish}", flush=True)
        except Exception as e:
            print(f"  LLM error (попытка {attempt+1}): {e}", flush=True)

    t5 = _time.time()
    print(f"  LLM ответ: {len(answer or '')} симв. ({t5-t4:.1f}с)", flush=True)
    print(f"  Всего: {t5-t0:.1f}с", flush=True)

    if not answer:
        # LLM-free fallback: извлекаем "ЗНАК — ЗОНА" из результатов и строим ответ
        SIGN_PLANET = {
            "овен": "Марс",
            "телец": "Венера",
            "близнецы": "Меркурий",
            "рак": "Луна",
            "лев": "Солнце",
            "дева": "Меркурий",
            "весы": "Венера",
            "скорпион": "Плутон",
            "стрелец": "Юпитер",
            "козерог": "Сатурн",
            "водолей": "Уран",
            "рыбы": "Нептун",
        }
        best_match = None
        # Берём sentence-level результаты (они точнее по смыслу)
        for r in sentence_results:
            txt = (r[0] if isinstance(r, tuple) and len(r) > 0 else str(r)).lower()
            for sign, planet in SIGN_PLANET.items():
                m = re.search(rf"{sign}\s*—\s*[^.]{{0,200}}?плит", txt)
                if m:
                    start = max(0, m.start() - 20)
                    ctx = txt[start : m.end() + 30]
                    best_match = (sign.capitalize(), planet, ctx)
                    break
            if best_match:
                break
        if best_match:
            answer = f"По данным источников, за плиту отвечает планета {best_match[1]} (через знак {best_match[0]})."
            print(f"  LLM-free fallback: {best_match[0]} \u2192 {best_match[1]}", flush=True)
        else:
            return combined[:top_k]

    result_text = (
        f"=== ОТВЕТ ===\n\n{answer}\n\n"
        f"=== МЕТОД ===\n"
        f"DAT Sentence-level + LLM | alpha={alpha} (sentence={s_score}/5, dense={d_score}/5)\n"
        f"Время: {t5-t0:.1f}с\n\n"
        f"=== ИСТОЧНИКИ ===\n{chunks_text[:3000]}"
    )
    return [(result_text, 999.0, 0, 0, 0)]


def search_self_rag(hipporag, query, top_k=30, candidate_pool=200):
    """
    Self-RAG: BM25 + dense (RRF) → LLM формулирует ответ.

    Раньше кандидатов отбирал чистый BM25 (P1-6): на вопрос «какие книги
    написала Максимова И.В.?» он поднимал прозу корпуса («например, написала
    я книгу») выше титульных страниц, потому что в них нет слов вопроса.
    Теперь кандидаты — объединение BM25-топа и dense-топа, а ранжирование
    считается по RRF: при отсутствии эмбеддингов поведение остаётся прежним.
    """
    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
    _, q_lower = split_query_terms(query)
    terms, _ = split_query_terms(query)

    chunk_store = hipporag.chunk_embedding_store
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    exp = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + exp))
    bm25_scores = np.asarray(bm25.get_scores(q_tokens))
    emb_scores = _embedding_scores(hipporag, chunk_store, query)

    # Шаг 1: кандидаты = BM25-топ ∪ dense-топ, ранжирование RRF
    if emb_scores is None or len(emb_scores) != n_chunks:
        candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]
        ranked = [(int(idx), float(bm25_scores[idx])) for idx in candidate_idx]
    else:
        bm25_pos = {int(idx): pos for pos, idx in enumerate(np.argsort(bm25_scores)[::-1])}
        emb_pos = {int(idx): pos for pos, idx in enumerate(np.argsort(emb_scores)[::-1])}
        pool = set(np.argsort(bm25_scores)[::-1][:candidate_pool].tolist())
        pool |= set(np.argsort(emb_scores)[::-1][:candidate_pool].tolist())
        k = 60
        ranked = [
            (idx, 1.0 / (k + bm25_pos.get(idx, 1e6)) + 1.0 / (k + emb_pos.get(idx, 1e6)))
            for idx in pool
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)

    scored = []
    for idx, base in ranked:
        tl = chunk_store.texts[idx].lower()
        boost = _keyword_boost(tl, q_lower, terms) + _chunk_sign_score(chunk_store, idx, q_lower)
        final = base * (1.0 + boost)
        scored.append((chunk_store.texts[idx], final, boost, idx))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Контекстное окно: добавляем соседние чанки
    used_ranges = set()
    merged = []
    for text, base_score, boost, idx in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for off in range(-1, 2):
            ni = idx + off
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= 1:
                window.add(ni)
        pages = set()
        for wi in window:
            _, wp = _get_chunk_meta(chunk_store, wi)
            pages.add(wp)
        mp, xp = min(pages), max(pages)
        rk = f"{source}:{mp}-{xp}"
        if rk in used_ranges:
            continue
        used_ranges.add(rk)
        parts = [chunk_store.texts[wi] for wi in sorted(window)]
        merged.append(("\n\n".join(parts), source, mp, xp))
        if len(merged) >= top_k:
            break

    # Шаг 2: собираем текст чанков для LLM
    chunks_text = ""
    for i, (text, source, mp, xp) in enumerate(merged, 1):
        chunks_text += f"\n[Источник {i} — {source} стр.{mp}-{xp}]\n{text[:1200]}\n"

    # Шаг 3: LLM
    answer = None
    try:
        r_health = httpx.get(f"{api_url}/v1/models", timeout=3)
        if r_health.status_code == 200:
            prompt = {
                "model": LLM_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ответь на вопрос читателя, используя ТОЛЬКО текст источников.\n\n"
                            "АЛГОРИТМ:\n"
                            "1. Вопрос может быть ЛЮБОЙ темы: автор, книга, издание, понятие, "
                            "знак зодиака, зона дома. Определи тип вопроса и найди прямые ответы "
                            "в источниках.\n"
                            "2. Для вопросов вида «какая планета отвечает за зону» — "
                            "ищи «знаку соответствует зона» + «управитель знака — планета»\n"
                            "3. Для остальных вопросов — приведи прямое содержание найденного: "
                            "названия, формулировки, факты, дословно как в источниках\n"
                            "4. Если в источниках есть нужная информация, но она сформулирована "
                            "ИНАЧЕ, чем вопрос (например «Психологический фэншуй» вместо "
                            "«название книги») — приведи её как ответ. Не считай это отсутствием "
                            "информации.\n"
                            "5. Для вопросов «перечисли всё» — выведи полный список\n"
                            "6. Ссылайся на источники: [Источник 1]\n"
                            "7. «нет данных» пиши ТОЛЬКО если в источниках действительно нет "
                            "ничего по теме вопроса. Отсутствие ГОТОВОЙ формулировки "
                            "не равно отсутствию информации: если в источниках есть нужные "
                            "слова, факты и соответствия, из которых ответ выводится — "
                            "составь и дай ответ, а не отказ"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Вопрос: {query}\n\n"
                            f"Источники:\n{chunks_text}\n\n"
                            f"Ответь на основе источников."
                        ),
                    },
                ],
                "max_tokens": 2048,
                "temperature": 0.15,
            }
            resp = httpx.post(f"{api_url}/v1/chat/completions", json=prompt, timeout=300)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                answer = (msg.get("content") or "").strip()
                if not answer and msg.get("reasoning_content"):
                    rc = msg["reasoning_content"]
                    lines = [l.strip() for l in rc.split("\n") if l.strip()]
                    answer = lines[-1].strip("\"'").replace("**", "") if lines else answer
    except Exception:
        pass

    if not answer:
        return [(scored[0][0], scored[0][1], "", 0, 0, 0)]

    result_text = f"=== ОТВЕТ ===\n\n{answer}\n\n=== ИСТОЧНИКИ ===\n{chunks_text[:3000]}"
    return [(result_text, 999.0, 0, 0, 0)]


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
                    "model": LLM_NAME,
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
                "model": LLM_NAME,
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


def search_filtered(hipporag, query, top_k=10, candidate_pool=200):
    """
    Small-to-big + RU-CE с фильтрацией по разделу знака зодиака.
    Если запрос про Тельца — показываем ТОЛЬКО чанки из раздела Тельца.
    """
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []

    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]

    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()

    expanded_terms = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + expanded_terms))
    bm25_scores = bm25.get_scores(q_tokens)

    # BM25 candidates → фильтрация по разделу
    candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]
    candidate_idx = _filter_by_section(candidate_idx, chunk_store, q_lower)

    if len(candidate_idx) == 0:
        return []

    scored = []
    for idx in candidate_idx:
        if idx >= n_chunks:
            continue
        text = chunk_store.texts[idx]
        tl = text.lower()
        emb = float(emb_scores[idx])
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final = emb * (1.0 + boost)
        scored.append((idx, final, emb, float(bm25_scores[idx]), boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    # RU Cross-encoder rerank top 30
    try:
        ce = _get_crossencoder_ru()
        top_30 = scored[:30]
        pairs = [(query, chunk_store.texts[idx][:512]) for idx, *_ in top_30]
        xc_scores = ce.predict(pairs, show_progress_bar=False)
        for i, (idx, final, emb, bm25_s, boost) in enumerate(top_30):
            xc = float(xc_scores[i])
            final = emb * 0.8 * (1.0 + boost) + xc * 0.2
            scored[i] = (idx, final, emb, bm25_s, boost, xc)
    except Exception:
        pass

    scored.sort(key=lambda x: x[1], reverse=True)

    # Small-to-big merging
    used_ranges = set()
    merged = []
    for idx, final, emb, bm25_s, boost, *_ in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-2, 3):
            ni = idx + offset
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= 2:
                window.add(ni)
        pages = set()
        for wi in window:
            _, wp = _get_chunk_meta(chunk_store, wi)
            pages.add(wp)
        min_p, max_p = min(pages), max(pages)
        rkey = f"{source}:{min_p}-{max_p}"
        if rkey in used_ranges:
            continue
        used_ranges.add(rkey)
        parts = [chunk_store.texts[wi] for wi in sorted(window)]
        full_text = "\n\n".join(parts)

        # Обрезка до заголовка знака
        sign_in_result = _sign_in_query(q_lower)
        if sign_in_result:
            import re as _re

            for form in ZODIAC_FORMS.get(sign_in_result, [sign_in_result]):
                pattern = rf"(?:^|[.!\n])\s*{_re.escape(form.upper())}(?:\s|[:,.!])"
                m = _re.search(pattern, full_text, _re.IGNORECASE)
                if m:
                    cut_pos = m.start()
                    if cut_pos > 0:
                        full_text = full_text[cut_pos:]
                    full_text = full_text.lstrip(" .\n\t,;:!?")
                    break

        boost_final = _keyword_boost(full_text.lower(), q_lower, terms)
        final_score = final * (1.0 + boost_final * 0.5)
        merged.append((full_text, final_score, source, min_p, max_p, boost_final))
        if len(merged) >= top_k:
            break

    # Дедупликация
    seen_texts = set()
    deduped = []
    for text, score, source, min_p, max_p, boost in merged:
        key = text[:200].lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped.append((text, score, source, min_p, max_p, boost))

    deduped.sort(key=lambda x: x[1], reverse=True)
    return deduped[:top_k]


def search_hyde(hipporag, query, top_k=10, candidate_pool=200):
    """
    HyDE (Hypothetical Document Embedding).
    1. LLM генерирует гипотетический документ (стиль книги Астрофэншуй)
    2. Эмбеддинг запроса + эмбеддинг гипотетического документа
    3. BM25 отбирает кандидатов
    4. Multi-vector scoring + sign dominance
    5. Русский Cross-encoder reranking
    """
    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL

    # Шаг 1: генерация гипотетического документа
    hyde_text = None
    try:
        r_health = httpx.get(f"{api_url}/v1/models", timeout=3)
        if r_health.status_code == 200:
            hyde_prompt = {
                "model": LLM_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты — автор книги «Астрофэншуй квартиры». Напиши отрывок из книги, "
                            "который отвечает на вопрос читателя. Используй стиль книги: "
                            "астрологические термины, фэншуй, знаки зодиака, планеты, зоны квартиры. "
                            "Напиши 3-5 предложений как в настоящей книге."
                        ),
                    },
                    {"role": "user", "content": f"Вопрос читателя: {query}"},
                ],
                "max_tokens": 1024,
                "temperature": 0.3,
            }
            r = httpx.post(f"{api_url}/v1/chat/completions", json=hyde_prompt, timeout=300)
            if r.status_code == 200:
                data = r.json()
                msg = data["choices"][0]["message"]
                hyde_text = (msg.get("content") or "").strip()
                if not hyde_text and msg.get("reasoning_content"):
                    rc = msg["reasoning_content"]
                    lines = [l.strip() for l in rc.split("\n") if l.strip()]
                    hyde_text = lines[-1].strip("\"'").replace("**", "") if lines else None
                if hyde_text and len(hyde_text) > 20:
                    print(f"  📄 Гипотетический документ сгенерирован ({len(hyde_text)} симв.)")
                else:
                    hyde_text = None
    except Exception:
        pass

    if not hyde_text:
        print("  ⚠️ HyDE: LLM недоступен, работаю без гипотетического документа")
        hyde_text = query

    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []

    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]

    # Эмбеддинг запроса и гипотетического документа
    q_emb = hipporag.embedding_model.encode([query])
    hyde_emb = hipporag.embedding_model.encode([hyde_text])
    all_emb = np.array(chunk_store.embeddings)

    query_scores = np.dot(all_emb, q_emb.T).squeeze()
    hyde_scores = np.dot(all_emb, hyde_emb.T).squeeze()

    # BM25 candidates
    q_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)
    candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]

    scored = []
    for idx in candidate_idx:
        if idx >= n_chunks:
            continue
        text = chunk_store.texts[idx]
        tl = text.lower()
        qs = float(query_scores[idx])
        hs = float(hyde_scores[idx])
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        # Комбинируем: query (40%) + hyde (40%) + keyword+sign (20%)
        final = qs * 0.4 + hs * 0.4 + (qs + hs) * 0.5 * boost * 0.2
        scored.append((text, final, qs, hs, boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Rerank с русским CE
    top_50 = scored[:50]
    rest = scored[50:]
    try:
        ce = _get_crossencoder_ru()
        pairs = [(query, c[0][:512]) for c in top_50]
        xc_scores = ce.predict(pairs, show_progress_bar=False)
        reranked = []
        for i, c in enumerate(top_50):
            xc = float(xc_scores[i])
            text = c[0]
            old = c[1]
            boost = c[4]
            final = old * 0.6 + xc * 0.4
            reranked.append((text, final, xc, c[2], boost))
        reranked.sort(key=lambda x: x[1], reverse=True)
        scored = reranked + rest
    except Exception:
        pass

    return scored[:top_k], hyde_text


def show_results(query, results, mode_name, reformulated=None):
    """Display search results with sentence-boundary detection."""
    q_lower = query.lower()
    print(f"\n{'=' * 80}")
    print(f"[{mode_name}] Результаты: '{query}'")
    if reformulated and reformulated != query:
        label = "гипотетический документ" if mode_name == "HyDE" else "переформулировка"
        print(f"  ({label}: '{reformulated[:120]}')")
    print(f"{'=' * 80}")
    if not results:
        print("Ничего не найдено.")
        return

    is_full_material = mode_name in ("Весь знак", "Резюме", "Self-RAG")

    for i, r in enumerate(results, 1):
        is_selfrag = mode_name == "Self-RAG"
        is_small_to_big = len(r) == 6 and isinstance(r[2], str) and not is_selfrag
        if is_small_to_big:
            doc, score, source, min_p, max_p, boost = r
            extra = f" ({str(source)[:40]} стр.{min_p}-{max_p})"
        else:
            doc, score, dpr, multi, boost = r
            extra = f" (dpr={dpr:.3f}, multi={multi:.3f})"

        has_term = q_lower in doc.lower()
        marker = " ✅" if has_term else ""

        if is_full_material:
            # Для полного материала: показываем резюме + путь к файлу
            display = doc
            # Если есть строка "Файл: " — выделяем её
            if "Файл:" in doc:
                for ln in doc.split("\n"):
                    if ln.startswith("Файл:") or "outputs/reports" in ln:
                        display = display.replace(ln, f"\n>>> {ln} <<<")
        else:
            clean_doc, was_truncated = fix_chunk_start(doc)
            display = show_text_safe(clean_doc)
            if was_truncated:
                display = "[...обрыв] " + display

        print(f"\n--- {i}. (score: {score:.4f}, kw: +{boost:.2f}){extra}{marker} ---")
        print(display)
        print()


MODEL_OPTIONS = {
    "1": ("google/gemma-4-e4b", "Gemma-4 ~9B активных (текущая)"),
    "2": ("google/gemma-4-26b-a4b", "Gemma-4 26B MoE ~4B активных, больше знаний"),
    "3": (
        "qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled@iq4_xs",
        "Qwen3.6 35B MoE Claude-distilled, быстрый",
    ),
    "4": ("nvidia/nemotron-3-nano-4b", "Nvidia Nemotron 4B, лёгкая и быстрая"),
}


def main():
    print("=" * 80)
    print("HIPPORAG - Поиск по базе знаний (астрофэншуй)")
    print("=" * 80)

    global LLM_NAME

    print("\nВыберите модель для ответов:")
    for k, v in MODEL_OPTIONS.items():
        cur = " ← текущая" if v[0] == LLM_NAME else ""
        print(f"  {k} — {v[1]}{cur}")
    model_choice = input("Выбор [1-4], Enter для текущей: ").strip()
    chosen_model = MODEL_OPTIONS.get(model_choice, (None, None))[0] or LLM_NAME
    if chosen_model != LLM_NAME:
        print(f"  → Переключаю API на: {chosen_model}")
    else:
        print(f"  → Модель: {chosen_model}")

    config = BaseConfig(
        save_dir=SAVE_DIR,
        llm_name=LLM_NAME,
        llm_base_url=LLM_BASE_URL,
        embedding_model_name=EMBEDDING_MODEL,
        linking_top_k=100,
        retrieval_top_k=200,
        passage_node_weight=0.05,
        embedding_batch_size=8,
    )

    print("\nЗагрузка индекса...")
    hipporag = HippoRAG(global_config=config)
    print("Готово!\n")

    if chosen_model != LLM_NAME:
        LLM_NAME = chosen_model

    while True:
        print("-" * 80)
        print("Режимы поиска:")
        print("  1 — Multi-vector + LLM переформулировка ⭐")
        print("  2 — Графовый (entity linking + PPR)")
        print("  3 — BM25 + Embedding + CE (English)")
        print("  4 — Small-to-big + CE (English)")
        print("  5 — BM25 + RU-CE + Query expansion")
        print("  6 — Small-to-big + RU-CE (русский)")
        print("  7 — HyDE + RU-CE (гипотетический документ) ⭐")
        print("  8 — Фильтр по разделу + RU-CE")
        print("  9 — Весь материал по знаку зодиака (полностью)")
        print("  10 — Весь материал + резюме LLM")
        print("  11 — Весь материал + LLM обработка (стиль автора)")
        print("  12 — Self-RAG (поиск + ответ LLM) ⭐")
        print("  13 — RRF + контекстное окно + RU-CE")
        print("  14 — E5 sentence-rerank")
        print("  15 — RRF + контекст + sign dominance")
        print("  16 — ColBERT-v2 late interaction")
        print("  17 — Любой метод + LLM фильтрация")
        print("  18 — Qwen3 + FRIDA + BM25 + LLM")
        print("  19 — GraphRAG Agent (маршрутизация + самокоррекция) ⭐")
        print("  20 — Sentence-level chunking (BM25 по предложениям) ⭐")
        print("       (названия книг/автор — это не предложение, см. режимы 2 и 5)")
        print("  21 — DAT Sentence-level + LLM (динамический α) ⭐")
        print("  22 — Pattern Search: знак ↔ зона (зона или знак из запроса)")
        print("  23 — QA Cache (кэш ответов; без кэша — режим 22)")
        print("  t — Тестирование всех методов по кэшу")
        print("  e — Экспорт кэша в JSONL (для обучения)")
        print("  q — Выход")
        mode = input("Выбор [1-23/q/t/e]: ").strip().lower()
        if mode == "q":
            break
        if mode == "t":
            q = input("Запрос для теста (Enter = из кэша): ").strip()
            if not q:
                cache = _qa_load()
                if cache:
                    q = cache[-1]["query"]
                    print(f"  Беру последний из кэша: {q}")
                else:
                    print("  Кэш пуст.")
                    continue
            _qa_run_test(hipporag, q)
            continue
        if mode == "e":
            _qa_export_training()
            continue
        if mode not in (
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
        ):
            continue

        query = input("Запрос: ").strip()
        if not query:
            continue

        reformulated = None
        hyde_text = None
        mode_map = {
            "1": ("Multi-vector", lambda q=query: search_multi_vector(hipporag, q), True),
            "2": ("Граф", lambda q=query: search_hipporag_graph(hipporag, q), False),
            "3": ("BM25+CE-en", lambda q=query: search_bm25_hybrid(hipporag, q), False),
            "4": ("Small→Big-en", lambda q=query: search_small_to_big(hipporag, q), False),
            "5": ("BM25+CE-ru", lambda q=query: search_bm25_hybrid_ru(hipporag, q), False),
            "6": ("Small→Big-ru", lambda q=query: search_small_to_big_ru(hipporag, q), False),
            "7": ("HyDE", lambda q=query: search_hyde(hipporag, q), True),
            "8": ("Фильтр", lambda q=query: search_filtered(hipporag, q), False),
            "9": ("Весь знак", lambda q=query: extract_sign_full(hipporag, q), False),
            "10": ("Резюме", lambda q=query: extract_sign_summary(hipporag, q), False),
            "11": ("Обраб.", lambda q=query: extract_sign_polished(hipporag, q), False),
            "12": ("Self-RAG", lambda q=query: search_self_rag(hipporag, q), False),
            "13": ("RRF+Контекст", lambda q=query: search_rrf_context(hipporag, q), False),
            "14": ("E5-rerank", lambda q=query: search_e5_rerank(hipporag, q), False),
            "15": ("RRF+sign", lambda q=query: search_mxbai(hipporag, q), False),
            "16": ("ColBERT", lambda q=query: search_colbert(hipporag, q), False),
            "17": ("LLM-фильтр", lambda q=query: search_with_llm_filter(hipporag, q), False),
            "18": ("Qwen+FRIDA", lambda q=query: search_qwen_frida(hipporag, q), False),
            "19": ("GraphRAG", lambda q=query: search_graphrag_agent(hipporag, q), False),
            "20": ("Sentence-level", lambda q=query: search_sentence_level(hipporag, q), False),
            "21": ("DAT+LLM", lambda q=query: search_dat_sentence_llm(hipporag, q), False),
            "22": ("SemFilter", lambda q=query: search_semantic_filter(hipporag, q), False),
            "23": ("QACache", lambda q=query: search_qa_cache(hipporag, q), False),
        }
        mode_name, search_fn, has_extra = mode_map[mode]
        if has_extra:
            result = search_fn()
            if mode == "1":
                results, reformulated = result
            else:
                results, hyde_text = result
        else:
            results = search_fn()

        # Автосохранение в QA Cache
        if results and mode not in ("23",):
            try:
                r0 = results[0]
                txt = r0[0] if isinstance(r0, tuple) and len(r0) > 0 else str(r0)
                # Честный отказ не кэшируем: после правок/переиндексации он
                # может перестать быть верным (P1-9).
                if "Ответ не найден" in txt:
                    raise ValueError("отказ не кэшируем")
                short = ""
                for line in txt.split("\n"):
                    if "ОТВЕТ:" in line:
                        short = line.split("ОТВЕТ:")[1].strip()
                        break
                sources = [l.strip() for l in txt.split("\n") if "Источник" in l[:20]]
                _qa_store(query, txt, short, mode_name, int(mode), sources[:5])
            except Exception:
                pass

        show_results(query, results, mode_name, reformulated if reformulated else hyde_text)

    print("\nДо свидания!")


# Планеты-управители знаков (P1-8).
SIGN_PLANET = {
    "овен": "Марс",
    "телец": "Венера",
    "близнецы": "Меркурий",
    "рак": "Луна",
    "лев": "Солнце",
    "дева": "Меркурий",
    "весы": "Венера",
    "скорпион": "Плутон",
    "стрелец": "Юпитер",
    "козерог": "Сатурн",
    "водолей": "Уран",
    "рыбы": "Нептун",
}

# Стемы зон в запросе → каноническое название зоны.
ZONE_ALIASES = {
    "плит": "плита",
    "плиту": "плита",
    "кладовк": "кладовка",
    "кладовку": "кладовка",
    "спальн": "спальня",
    "спальню": "спальня",
    "ванн": "ванная",
    "ванну": "ванная",
    "кухн": "кухня",
    "кухню": "кухня",
    "балкон": "балкон",
    "коридор": "коридор",
    "гостин": "гостиная",
    "гостиную": "гостиная",
    "прихож": "прихожая",
    "прихожую": "прихожая",
    "туалет": "туалет",
    "шкаф": "шкаф",
    "двер": "дверь",
}

# Формы, которыми те же зоны называются в тексте корпуса (не в запросе).
_ZONE_TEXT_STEMS = {
    **ZONE_ALIASES,
    "холл": "прихожая",
    "лоджи": "балкон",
    "террас": "балкон",
    "гардероб": "кладовка",
    "кладовая": "кладовка",
}

# Маркеры, которыми корпус связывает знак с зоной.
_SIGN_ZONE_STRONG = ("обител", "соответству", "точка силы", "приписыва")
_SIGN_ZONE_WEAK = ("экзальтац", "изгнан", "падени", "управител", "зона")

_CLASS_NAMES = {0: "светская обитель", 1: "экзальтация", 2: "изгнание/падение"}

_ACCUSATIVE = {
    "плита": "плиту",
    "кладовка": "кладовку",
    "спальня": "спальню",
    "ванная": "ванную",
    "кухня": "кухню",
    "гостиная": "гостиную",
    "прихожая": "прихожую",
    "кладовая": "кладовую",
    "комната": "комнату",
    "терраса": "террасу",
    "лоджия": "лоджию",
    "дверь": "дверь",
    "квартира": "квартиру",
}

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


# ═══════════════════════════════════════════════════════════════
# QA Cache: база вопросов-ответов (кэш + тесты + обучение)
# ═══════════════════════════════════════════════════════════════

QA_CACHE_PATH = Path("outputs/qa_cache/qa_cache.json")

# Версия формата записи и логики ответа (P1-9). Кэш переживал правки кода:
# записи, сохранённые до фиксов, отдавались как «кэшированный ответ», то есть
# неверный ответ под видом кэша. Увеличивайте версию при изменении формата
# ответа/промптов — старые записи перестанут читаться и будут пересчитаны.
QA_CACHE_VERSION = 2


def _qa_is_current(entry):
    """Запись актуальна: совпадает версия формата и модель."""
    if not isinstance(entry, dict):
        return False
    if entry.get("schema_version") != QA_CACHE_VERSION:
        return False
    model = entry.get("model")
    if model and model != LLM_NAME:
        return False
    return True


def _qa_load():
    if QA_CACHE_PATH.exists():
        try:
            with open(str(QA_CACHE_PATH), "r", encoding="utf-8") as f:
                cache = json.load(f)
            if isinstance(cache, list):
                return [entry for entry in cache if _qa_is_current(entry)]
        except Exception:
            pass
    return []


def _qa_save(cache):
    QA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(QA_CACHE_PATH), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _qa_norm(q):
    return re.sub(r"[^\w\s]", "", q.lower()).strip()


def _qa_lookup(query, cache):
    """Поиск похожего вопроса в кэше. Возвращает (entry, score) или None."""
    qn = _qa_norm(query)
    q_words = set(qn.split())
    best = None
    best_score = 0
    for e in cache:
        en = e.get("query_norm", "")
        e_words = set(en.split())
        if not q_words or not e_words:
            continue
        overlap = len(q_words & e_words) / max(len(q_words), len(e_words))
        if overlap > best_score:
            best_score = overlap
            best = e
    if best_score >= 0.7:
        return best, best_score
    return None, 0


def _qa_derive_short(text: str) -> str:
    """Короткий ответ из текста результата, если нет маркера 'ОТВЕТ:'.

    Раньше для режимов без маркера (например 22 — SemFilter) в кэш попадал
    пустой short_answer, и режим 23 отдавал «ОТВЕТ: не указан» (P2-8).
    """
    if not text:
        return ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("===") and line.endswith("==="):
            continue
        if line.startswith("["):  # строки вида "[Источник 1 — file.pdf стр.1-2]"
            continue
        return line[:200]
    return ""


def _qa_store(query, answer, short_answer, method_name, method_id, sources, score=999.0):
    """Сохранить результат в кэш."""
    cache = _qa_load()
    qn = _qa_norm(query)
    # Не дублировать точные совпадения
    for e in cache:
        if e.get("query_norm") == qn and e.get("method_id") == method_id:
            return
    if not short_answer:
        short_answer = _qa_derive_short(answer)
    entry = {
        "id": f"qa_{len(cache) + 1}",
        "schema_version": QA_CACHE_VERSION,
        "query": query,
        "query_norm": qn,
        "answer": answer if answer else "",
        "short_answer": short_answer[:200] if short_answer else "",
        "method": method_name,
        "method_id": method_id,
        "model": LLM_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": sources[:10] if sources else [],
        "score": score,
    }
    cache.append(entry)
    # _qa_load уже отбросил записи старой версии/модели, поэтому сохранение
    # заодно вычищает устаревшие записи с диска (P1-9).
    _qa_save(cache)
    print(f"  Сохранено в QA Cache (#{len(cache)})", flush=True)


def _qa_export_training(output_path="outputs/qa_cache/training_data.jsonl"):
    """Экспорт кэша в JSONL для fine-tuning."""
    cache = _qa_load()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out), "w", encoding="utf-8") as f:
        for e in cache:
            line = json.dumps(
                {
                    "instruction": e["query"],
                    "output": e["short_answer"] or e["answer"][:500],
                    "sources": e["sources"],
                    "method": e["method"],
                },
                ensure_ascii=False,
            )
            f.write(line + "\n")
    print(f"  Экспортировано {len(cache)} записей в {out}", flush=True)


def _qa_run_test(hipporag, query):
    """Запустить все методы и сравнить с кэшем."""
    from main import (
        search_dat_sentence_llm,
        search_self_rag,
        search_semantic_filter,
        search_sentence_level,
        search_small_to_big_ru,
    )

    test_methods = [
        (6, "Small→Big-ru", lambda: search_small_to_big_ru(hipporag, query)),
        (12, "Self-RAG", lambda: search_self_rag(hipporag, query)),
        (20, "Sentence", lambda: search_sentence_level(hipporag, query)),
        (21, "DAT+LLM", lambda: search_dat_sentence_llm(hipporag, query)),
        (22, "Pattern", lambda: search_semantic_filter(hipporag, query)),
    ]
    cache = _qa_load()
    cached, _ = _qa_lookup(query, cache)
    print(f"\n{'='*60}")
    print("ТЕСТИРОВАНИЕ ВСЕХ МЕТОДОВ")
    print(f"Запрос: {query}")
    if cached:
        print(f"Ожидаемый ответ: {cached.get('short_answer', '')[:100]}")
    print(f"{'='*60}\n")
    print(f"{'Метод':<15} {'Ответ (первые 80)':<80} {'Совпадение':<10}")
    print("-" * 105)
    for mid, mname, mfunc in test_methods:
        try:
            res = mfunc()
            txt = ""
            if res:
                r = res[0]
                txt = r[0] if isinstance(r, tuple) and len(r) > 0 else str(r)
            short = txt[:80].replace("\n", " ").strip()
            match = "✓" if cached and cached.get("short_answer", "") in txt else "?"
            print(f"{mname:<15} {short:<80} {match:<10}")
        except Exception as e:
            print(f"{mname:<15} {'ERROR: ' + str(e)[:60]:<80} {'':<10}")


def search_qa_cache(hipporag, query, top_k=10):
    """
    Метод 23: QA Cache — поиск по базе готовых ответов.
    Если вопрос уже задавался — возвращает кэшированный ответ.
    Если нет — запускает метод 22, сохраняет результат, возвращает.
    """
    import time as _time

    t0 = _time.time()
    print("  Метод 23: QA Cache", flush=True)
    cache = _qa_load()
    cached, score = _qa_lookup(query, cache)
    if cached:
        print(f"  Найден в кэше (совпадение: {score:.0%})", flush=True)
        short = cached.get("short_answer", "")
        full = cached.get("answer", "")
        # Записи, созданные до P2-8, могли остаться с пустым short_answer —
        # восстанавливаем короткий ответ из полного (P2-8).
        if not short:
            short = _qa_derive_short(full)
        result = (
            f"=== ОТВЕТ ИЗ КЭША ===\n"
            f"Запрос: {query}\n"
            f"Совпадение: {score:.0%}\n\n"
            f"ОТВЕТ: {short or 'не указан'}\n\n"
            f"--- Полный ответ ---\n"
            f"{full}\n\n"
            f"Сохранён: {cached['timestamp']} | Метод: {cached['method']} | Модель: {cached['model']}"
        )
        print(f"  Готово за {_time.time() - t0:.2f}с", flush=True)
        return [(result, 999.0, 0, 0, 0)]
    # Не найдено → запускаем метод 22, сохраняем
    print("  Не найден в кэше, запускаю метод 22...", flush=True)
    res22 = search_semantic_filter(hipporag, query, top_k=50)
    if res22 and len(res22) > 0:
        txt = res22[0][0] if isinstance(res22[0], tuple) else str(res22[0])
        # Честный отказ не кэшируем: после переиндексации/правок он может
        # перестать быть верным, а кэш обязан быть версионным (P1-9).
        if "Ответ не найден" in txt:
            print(f"  Готово за {_time.time() - t0:.2f}с (отказ не кэшируется)", flush=True)
            return res22
        # Извлекаем короткий ответ
        short = ""
        for line in txt.split("\n"):
            if "ОТВЕТ:" in line:
                short = line.split("ОТВЕТ:")[1].strip()
                break
        sources = []
        for line in txt.split("\n"):
            if line.startswith("[Источник"):
                sources.append(line.strip())
        _qa_store(query, txt, short, "QACache+Pattern", 23, sources, 999.0)
        print(f"  Готово за {_time.time() - t0:.2f}с", flush=True)
        return res22
    return [("Не удалось получить ответ.", 0, 0, 0, 0)]


if __name__ == "__main__":
    main()
