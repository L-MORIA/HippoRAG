# Быстрый старт HippoRAG (RU)

Пять шагов до первого ответа. Полное руководство — [USER_GUIDE.md](./USER_GUIDE.md),
установка и замеры — [HOW_TO_RUN.md](./HOW_TO_RUN.md).

## 1. Установка

Нужны Python 3.10–3.12 и 8+ ГБ RAM.

```bash
git clone https://github.com/L-MORIA/HippoRAG.git
cd HippoRAG
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 2. LM Studio

Запустите LM Studio → **Developer → Start Server** (порт `1234`) и проверьте:

```bash
curl http://localhost:1234/v1/models
```

Скопируйте `.env.example` в `.env`:

```dotenv
LLM_NAME=gemma-4-e4b
LLM_API_NAME=google/gemma-4-e4b
LLM_BASE_URL=http://localhost:1234/v1
SAVE_DIR=outputs/astrology_test
```

## 3. Индекс

Положите PDF в `data/pdfs/` и соберите индекс:

```bash
python index_pdfs.py --pdf_dir data/pdfs --output_dir outputs/astrology_test
python scripts/show_index_stats.py --save_dir outputs/astrology_test
```

`--output_dir` и `SAVE_DIR` должны совпадать. Без индекса поиск вернёт отказ.

## 4. Запуск

Веб-интерфейс:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:API_KEY = "your-secret-key"
uvicorn astro_app.web:app --host 127.0.0.1 --port 8000
```

Откройте `http://127.0.0.1:8000/` и введите ключ. Для локального запуска без
ключа: `$env:API_KEY_REQUIRED = "0"`.

CLI:

```bash
python main.py
```

Режим по умолчанию — **5 «BM25+CE-ru»**; рекомендованы также **2 «Граф»** и
**22 «Pattern Search»**.

## 5. Первый вопрос

Спросите, например: «Какие зоны соответствуют знаку Овна?». В ответе будут
ссылки `[n]` и список использованных цитат. Если ответа нет в источниках,
система честно откажется отвечать.

Если что-то не работает — [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).
