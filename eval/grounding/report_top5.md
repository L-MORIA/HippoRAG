# Заземление Q&A: отчёт golden set

- Дата: 2026-11-05T10:56:19
- Режим: `real-rag_qa`
- Разметка: `eval/grounding/golden_qa.jsonl`
- Запуск: `engine=HippoRAG.rag_qa, save_dir=outputs/astrology_test, llm_name=gemma-4-e4b, llm_api_name=google/gemma-4-e4b, llm_base_url=http://localhost:1234/v1, embedding_model_name=sentence-transformers/paraphrase-multilingual-mpnet-base-v2, qa_top_k=5, qa_strict_grounding=True`
- Вопросов: 36 (answerable 23, borderline 5, unanswerable 13)

## Метрики

| Метрика | Значение | Порог | Статус |
|---|---:|---:|---|
| Abstention rate (unanswerable) | 92.3% | >= 90.0% | ✅ |
| False answer rate | 7.7% | <= 10.0% | ✅ |
| Answer rate (answerable) | 34.8% | >= 90.0% | ❌ |
| Citation precision | 0.125 | >= 0.800 | ❌ |
| Citation recall | 0.125 | >= 0.800 | ❌ |
| Marker validity | 1.000 | >= 1.000 | ✅ |
| Uncited sentence rate | 0.0% | <= 10.0% | ✅ |
| Answer accuracy | 8.7% | >= 70.0% | ❌ |

Латентность: p50 21354.86 мс, p95 25416.95 мс.

## Вопросы

| id | answerable | ответ | цитаты P/R | accuracy | ошибки |
|---|---|---|---|---|---|
| `a01-astro-author-edition` | да | отказ | — | ❌ | — |
| `a02-psy-author` | да | отказ | — | ❌ | — |
| `a03-psy-subtitle` | да | отказ | — | ❌ | — |
| `a04-psy-year-pages` | да | отказ | — | ❌ | — |
| `a05-astro-isbn` | да | отказ | — | ❌ | — |
| `a06-eso-publisher-license` | да | ответ | 0.000 / 0.000 | ❌ | — |
| `a07-owen-zones` | да | ответ | 0.000 / 0.000 | ❌ | — |
| `a08-owen-secular` | да | отказ | — | ❌ | — |
| `a09-taurus-secular` | да | отказ | — | ❌ | — |
| `a10-scorpio-zones` | да | отказ | — | ❌ | — |
| `a11-dominant-zone` | да | отказ | — | ❌ | — |
| `a12-shen-qi` | да | ответ | 1.000 / 1.000 | ✅ | — |
| `a13-gua1-shen-qi` | да | ответ | 0.000 / 0.000 | ✅ | — |
| `a14-bagua-squares` | да | отказ | — | ❌ | — |
| `a15-house4` | да | отказ | — | ❌ | — |
| `a16-traditional-style` | да | ответ | 0.000 / 0.000 | ❌ | — |
| `a17-wealth-zone` | да | ответ | 0.000 / 0.000 | ❌ | — |
| `a18-meta-zodiac` | да | отказ | — | ❌ | — |
| `b01-owen-two-forms` | да | отказ | — | ❌ | — |
| `b02-shen-qi-gua1` | да | отказ | — | ❌ | — |
| `b03-scorpio-taurus` | да | ответ | 0.000 / 0.000 | ❌ | — |
| `b04-owen-planet-kitchen` | да | отказ | — | ❌ | — |
| `b05-ryby-spalnya` | да | ответ | 0.000 / 0.000 | ❌ | — |
| `u01-ophiuchus` | нет | отказ | — | — | — |
| `u02-gua-2026` | нет | отказ | — | — | — |
| `u03-qigong-owen` | нет | отказ | — | — | — |
| `u04-renovation-cost` | нет | отказ | — | — | — |
| `u05-bagua-apps` | нет | отказ | — | — | — |
| `u06-mercury-crypto` | нет | отказ | — | — | — |
| `u07-pickle-recipes` | нет | отказ | — | — | — |
| `u08-student-maria` | нет | отказ | — | — | — |
| `u09-consultation-prices` | нет | ответ | — | — | — |
| `u10-weather-2026` | нет | отказ | — | — | — |
| `u11-whale-sign` | нет | отказ | — | — | — |
| `u12-path-pages` | нет | отказ | — | — | — |
| `u13-car-number-gua` | нет | отказ | — | — | — |

## Как читать

- Citation precision/recall считаются по размеченным `gold_sources`; для ответа без цитат precision = 0.
- Answer accuracy — все ключевые термины эталона найдены в теле ответа. Это не LLM-судья и не проверка истинности.
- Marker validity = валидные `[n]` / все `[n]`; ответ без маркеров при наличии ответа даёт 0.
- Uncited sentence rate — доля ответов, где есть предложение длиннее 40 символов без `[n]`.
