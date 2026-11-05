# H2-6: замер заземления на golden set (2026-11-05)

> **Связанные документы:** [DEEPSEEK_PLAN_2026-11-04.md](./DEEPSEEK_PLAN_2026-11-04.md) §5 (этап 0), §9 (DoD), [ROADMAP_REVISION_DEEPSEEK.md](./ROADMAP_REVISION_DEEPSEEK.md) §9.3 (H2-6), [PERFORMANCE.md](./PERFORMANCE.md), [H1_3_LATENCY_REPORT_2026-11-04.md](./H1_3_LATENCY_REPORT_2026-11-04.md)
> **Дата:** 2026-11-05 · **Вопросов:** 36 (answerable 23, borderline 5, unanswerable 13)

## 1. Короткий итог

Механика заземления работает: модель не выдумывает номера источников
(marker validity 1,000), отказывается на 92,3 % unanswerable-вопросов и почти не
оставляет предложений без ссылок. Узкое место — **доставка доказательства в
контекст**: page recall@5 у графового ретрива HippoRAG 8,7 %, у продуктового
режима 5 — 56,5 %.

Из этого следуют два вывода:

1. Пороги H2-6 по answer rate (≥ 90 %) и accuracy (≥ 70 %) недостижимы
   настройкой промпта: при recall@5 56,5 % максимум answer rate — 56,5 %,
   остальное — работа над ретривом (область H1).
2. `qa_top_k` не лечит. Подъём 5 → 8 не улучшает ни один путь: у HippoRAG
   ложные ответы удваиваются, у режима 5 accuracy падает с 56,5 % до 52,2 %.

## 2. Что мерили

| Параметр | Значение |
|---|---|
| Golden set | `eval/grounding/golden_qa.jsonl`, 36 вопросов |
| Индекс | `outputs/astrology_test/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2` (1369 чанков, 653 страницы, 3 PDF) |
| LLM | LM Studio, `google/gemma-4-e4b` (каталог индекса `gemma-4-e4b`) |
| Пути | `HippoRAG.rag_qa` и продуктовый режим 5 (BM25 + RU-CE) |
| Контекст | `qa_top_k` = 5 и 8 фрагментов |

Метрики считает `scripts/eval_grounding.py`: отказ/ответ, citation precision и
recall по размеченным `gold_sources`, marker validity, доля предложений без
ссылки, accuracy по `gold_terms`, p50/p95 латентности.

## 3. Метрики заземления

| Метрика | HippoRAG @5 | HippoRAG @8 | Режим 5 @5 | Режим 5 @8 | Порог |
|---|---:|---:|---:|---:|---:|
| Answer rate (answerable) | 34,8 % | 34,8 % | **73,9 %** | **78,3 %** | ≥ 90 % |
| Abstention (unanswerable) | 92,3 % | 84,6 % | 92,3 % | 92,3 % | ≥ 90 % |
| False answer rate | 7,7 % | 15,4 % | 7,7 % | 7,7 % | ≤ 10 % |
| Citation precision | 0,125 | 0,156 | **0,667** | 0,602 | ≥ 0,80 |
| Citation recall | 0,125 | 0,188 | **0,765** | 0,750 | ≥ 0,80 |
| Marker validity | 1,000 | 1,000 | 1,000 | 1,000 | ≥ 1,00 |
| Uncited sentence rate | 0 % | 20,0 % | **0 %** | 10,5 % | ≤ 10 % |
| Accuracy | 8,7 % | 13,0 % | **56,5 %** | 52,2 % | ≥ 70 % |
| p50 латентность | 21,6 с | 10,2 с | **12,9 с** | 14,1 с | — |

Пройдено у режима 5 @5: abstention, false answer, marker validity, uncited
sentences. Не пройдено: answer rate, citation precision/recall, accuracy.

Отчёты: `eval/grounding/report_top5.*`, `report_top8.*` (HippoRAG),
`report_mode5.*`, `report_mode5_top8.*` (режим 5). Предсказания — рядом, в
`predictions_*.jsonl`.

## 4. Диагностика ретрива (23 answerable, без генерации)

Доля вопросов, у которых размеченная страница попала в контекст:

| Ретривер | @5 | @10 | @20 | @50 |
|---|---:|---:|---:|---:|
| HippoRAG graph (факты + PPR) | 8,7 % | 17,4 % | 30,4 % | 65,2 % |
| HippoRAG dense (DPR) | 21,7 % | 21,7 % | 34,8 % | 69,6 % |
| Режим 5 (BM25 + RU-CE) | **56,5 %** | **65,2 %** | — | — |

Три вывода:

* графовый PPR на этом корпусе **хуже** обычного dense: сущностные связи тянут
  соседние страницы и вытесняют нужную;
* продуктовый ретривер в 6,5 раза точнее графового при том же индексе;
* даже k = 50 упирается в 65–70 %, то есть «докрутить» `qa_top_k` до порога
  80 % нельзя в принципе.

Чанки не склеены между страницами (1369 чанков на 653 страницы, у каждого ровно
одна запись метаданных), поэтому метрика не занижена из-за границ чанков.

## 5. Где теряются ответы (режим 5, @5)

Из 23 answerable: 13 верных, 10 потерь.

| Тип потери | Кол-во | Вопросы |
|---|---:|---|
| Отказ: gold-страницы нет в контексте | 6 | `a04-psy-year-pages`, `a13-gua1-shen-qi`, `a15-house4`, `b02-shen-qi-gua1`, `b04-owen-planet-kitchen`, `b05-ryby-spalnya` |
| Ответ по чужой странице | 3 | `a10-scorpio-zones` (цитаты 45/49/181 вместо 94), `a14-bagua-squares` (221 вместо 3), `b03-scorpio-taurus` (20/45/97 вместо 94/109) |
| Верная цитата, неполный ответ | 1 | `a18-meta-zodiac` |
| Ложный ответ на unanswerable | 1 | `u11-whale-sign` (ответ про Рака и Хирона) |

`a18-meta-zodiac` — вопрос калибровки разметки, а не модели: цитата на стр. 160
верная, ответ («прокачать сферы деятельности с помощью ремонта квартиры, метод
мета зодиака») отвечает на заданный вопрос, но `gold_terms` требуют ещё
«солнце» и «асцендент», которых в ответе нет. Такие термины стоит либо убрать,
либо добавить в вопрос явный запрос про Солнце и Асцендент.

## 6. Решение по `qa_top_k`

Оставляем **5**:

* HippoRAG: 5 → 8 не меняет answer rate (34,8 %), удваивает ложные ответы
  (7,7 % → 15,4 %) и снимает нулевую долю предложений без ссылок (0 % → 20,0 %);
* режим 5: 5 → 8 добавляет один ответ (73,9 % → 78,3 %), но снижает accuracy
  (56,5 % → 52,2 %), citation precision (0,667 → 0,602) и даёт 10,5 %
  предложений без ссылок.

Правило плана «если page recall@5 < 80 %, поднять `qa_top_k` до 8–10» по факту
не сработало: рост контекста добавляет шум быстрее, чем доказательства.
Дальнейший рост качества — только через ретрив.

## 7. Что не достигнуто и почему

| DoD | Факт | Причина |
|---|---|---|
| Answer rate ≥ 90 % | 73,9 % | recall@5 = 56,5 %: 6 отказов из 10 потерь — gold-страницы не было в контексте |
| Accuracy ≥ 70 % | 56,5 % | те же 6 отказов + 3 ответа по чужой странице |
| Citation precision ≥ 0,80 | 0,667 | модель цитирует все использованные фрагменты, часть из них не gold |
| Citation recall ≥ 0,80 | 0,765 | то же |
| Abstention ≥ 90 % | 92,3 % | выполнено |
| False answer ≤ 10 % | 7,7 % | выполнено |
| Marker validity ≥ 1,0 | 1,000 | выполнено |
| Uncited ≤ 10 % | 0 % | выполнено |

## 8. Воспроизведение

```powershell
# HippoRAG.rag_qa
python scripts/grounding_retriever.py --save-dir outputs/astrology_test `
    --llm-name gemma-4-e4b --llm-api-name google/gemma-4-e4b `
    --llm-base-url http://localhost:1234/v1 --qa-top-k 5 `
    --out-json eval/grounding/report_top5.json `
    --out-md eval/grounding/report_top5.md `
    --out-predictions eval/grounding/predictions_top5.jsonl

# Продуктовый режим 5
$env:ASTRO_RETRIEVER_TOP_K = "5"
$env:LLM_API_NAME = "google/gemma-4-e4b"
python scripts/eval_grounding.py --run --retriever grounding_astro_retriever:retrieve `
    --out-json eval/grounding/report_mode5.json `
    --out-md eval/grounding/report_mode5.md `
    --out-predictions eval/grounding/predictions_mode5.jsonl
```

## 9. Что дальше

1. **P1 (H1): поднять page recall@5 до ≥ 80 %.** Это единственный рычаг,
   который двигает answer rate и accuracy. Кандидаты: page-level чанки или
   parent-document retrieval, query expansion по терминам книги, обученный
   RU-reranker поверх top-50.
2. Калибровка разметки: `a18-meta-zodiac` (лишние `gold_terms`), `a07-owen-zones`
   (три классификации зон Овна в книге; сейчас gold — стр. 7).
3. `HippoRAG.rag_qa` оставить экспериментальным путём: продуктовый режим 5
   даёт в 6,5 раза лучший recall при том же индексе.
