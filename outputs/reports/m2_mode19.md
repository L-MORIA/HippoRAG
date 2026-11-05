# Заземление Q&A: отчёт golden set

- Дата: 2026-11-05T20:55:01
- Режим: `predictions`
- Разметка: `eval/grounding/golden_qa.jsonl`
- Вопросов: 36 (answerable 23, borderline 5, unanswerable 13)

## Метрики

| Метрика | Значение | Порог | Статус |
|---|---:|---:|---|
| Abstention rate (unanswerable) | 92.3% | >= 90.0% | ✅ |
| False answer rate | 7.7% | <= 10.0% | ✅ |
| Answer rate (answerable) | 82.6% | >= 90.0% | ❌ |
| Citation precision | 0.122 | >= 0.800 | ❌ |
| Citation recall | 0.210 | >= 0.800 | ❌ |
| Marker validity | 1.000 | >= 1.000 | ✅ |
| Uncited sentence rate | 10.0% | <= 10.0% | ✅ |
| Answer accuracy | 47.8% | >= 70.0% | ❌ |

Латентность: p50 19945.4 мс, p95 32749.38 мс.

## Retrieval (H1-6)

| Метрика | Значение | Порог | Статус |
|---|---:|---:|---|
| Page recall@5 (H1-6) | 30.4% | >= 80.0% | ❌ |

Page recall@10: 30.4%, @50: 43.5%; page coverage@5: 26.1%.

## Вопросы

| id | answerable | ответ | page@5 | цитаты P/R | accuracy | ошибки |
|---|---|---|---|---|---|---|
| `a01-astro-author-edition` | да | ответ | ❌ | 0.000 / 0.000 | ✅ | — |
| `a02-psy-author` | да | ответ | ❌ | 0.000 / 0.000 | ✅ | — |
| `a03-psy-subtitle` | да | ответ | ❌ | 0.000 / 0.000 | ✅ | — |
| `a04-psy-year-pages` | да | отказ | ❌ | — | ❌ | — |
| `a05-astro-isbn` | да | ответ | ❌ | 0.000 / 0.000 | ✅ | — |
| `a06-eso-publisher-license` | да | ответ | ✅ | 1.000 / 1.000 | ✅ | — |
| `a07-owen-zones` | да | ответ | ❌ | 0.000 / 0.000 | ❌ | — |
| `a08-owen-secular` | да | ответ | ✅ | 0.000 / 0.000 | ✅ | — |
| `a09-taurus-secular` | да | ответ | ❌ | 0.000 / 0.000 | ✅ | — |
| `a10-scorpio-zones` | да | ответ | ❌ | 0.000 / 0.000 | ❌ | — |
| `a11-dominant-zone` | да | отказ | ✅ | — | ❌ | — |
| `a12-shen-qi` | да | ответ | ❌ | 0.000 / 0.000 | ✅ | — |
| `a13-gua1-shen-qi` | да | ответ | ❌ | 0.167 / 1.000 | ❌ | — |
| `a14-bagua-squares` | да | отказ | ❌ | — | ❌ | — |
| `a15-house4` | да | отказ | ❌ | — | ❌ | — |
| `a16-traditional-style` | да | ответ | ✅ | 0.000 / 0.000 | ❌ | — |
| `a17-wealth-zone` | да | ответ | ❌ | 0.000 / 0.000 | ❌ | — |
| `a18-meta-zodiac` | да | ответ | ❌ | 0.000 / 0.000 | ❌ | — |
| `b01-owen-two-forms` | да | ответ | ✅ | 0.500 / 0.500 | ✅ | — |
| `b02-shen-qi-gua1` | да | ответ | ✅ | 0.400 / 1.000 | ✅ | — |
| `b03-scorpio-taurus` | да | ответ | ❌ | 0.000 / 0.000 | ❌ | — |
| `b04-owen-planet-kitchen` | да | ответ | ❌ | 0.000 / 0.000 | ✅ | — |
| `b05-ryby-spalnya` | да | ответ | ✅ | 0.250 / 0.500 | ❌ | — |
| `u01-ophiuchus` | нет | отказ | — | — | — | — |
| `u02-gua-2026` | нет | отказ | — | — | — | — |
| `u03-qigong-owen` | нет | отказ | — | — | — | — |
| `u04-renovation-cost` | нет | отказ | — | — | — | — |
| `u05-bagua-apps` | нет | ответ | — | — | — | — |
| `u06-mercury-crypto` | нет | отказ | — | — | — | — |
| `u07-pickle-recipes` | нет | отказ | — | — | — | — |
| `u08-student-maria` | нет | отказ | — | — | — | — |
| `u09-consultation-prices` | нет | отказ | — | — | — | — |
| `u10-weather-2026` | нет | отказ | — | — | — | — |
| `u11-whale-sign` | нет | отказ | — | — | — | — |
| `u12-path-pages` | нет | отказ | — | — | — | — |
| `u13-car-number-gua` | нет | отказ | — | — | — | — |

## Как читать

- Citation precision/recall считаются по размеченным `gold_sources`; для ответа без цитат precision = 0.
- Answer accuracy — все ключевые термины эталона найдены в теле ответа. Это не LLM-судья и не проверка истинности.
- Marker validity = валидные `[n]` / все `[n]`; ответ без маркеров при наличии ответа даёт 0.
- Uncited sentence rate — доля ответов, где есть предложение длиннее 40 символов без `[n]`.
