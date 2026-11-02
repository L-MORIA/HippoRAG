# Fresh Audit: HippoRAG Codebase
**Дата:** 2026-11-01  
**Аудитор:** OpenCode Agent  
**Статус:** Критические проблемы обнаружены

---

## Executive Summary

Проект HippoRAG содержит ряд критических проблем, которые могут привести к crash'ам в production. Обнаружены:
- 2 случая `assert False` (garbage collection crash)
- 2 случая `ipdb.set_trace()` в production-коде
- 4 случая голого `except:`
- 37 случаев `print()` вместо логирования
- 2 случая `raise Exception()` вместо специализированных исключений

---

## Critical Issues (Must Fix)

### 1. CRITICAL: `assert False` в HippoRAG.py:249
**Файл:** `src/hipporag/HippoRAG.py`  
**Строка:** 249

```python
assert False, logger.info("Done with OpenIE, run online indexing for future retrieval.")
```

**Проблема:** `assert False` вызовет `AssertionError` при `save_openie=True`. Это мёртвый код, который должен быть удалён или заменён на нормальное завершение.

**Риск:** Crash приложения при использовании OpenIE с сохранением результатов.

**Рекомендация:** Заменить на:
```python
logger.info("Done with OpenIE, run online indexing for future retrieval.")
```

---

### 2. CRITICAL: `assert False` в embedding_model/__init__.py:38
**Файл:** `src/hipporag/embedding_model/__init__.py`  
**Строка:** 38

```python
assert False, f"Unknown embedding model name: {embedding_model_name}"
```

**Проблема:** При использовании неизвестной модели embedding произойдёт crash с `AssertionError`.

**Риск:** Приложение упадёт при попытке использовать нестандартную модель эмбеддингов.

**Рекомендация:** Заменить на корректное исключение:
```python
raise ValueError(f"Unknown embedding model name: {embedding_model_name}")
```

---

### 3. CRITICAL: `ipdb.set_trace()` в StandardRAG.py:103
**Файл:** `src/hipporag/StandardRAG.py`  
**Строка:** 103

```python
ipdb.set_trace()
```

**Проблема:** Отладочный код оставлен в production. При выполнении этой ветки код остановится и будет ждать ввода из терминала.

**Риск:** Hang приложения, блокировка потока выполнения.

**Рекомендация:** Удалить эту строку или заменить на логирование.

---

### 4. CRITICAL: `ipdb.set_trace()` в embedding_model/OpenAI.py:114
**Файл:** `src/hipporag/embedding_model/OpenAI.py`  
**Строка:** 114

```python
ipdb.set_trace()
```

**Проблема:** Отладочный код в production.

**Риск:** Hang приложения при выполнении этой ветки.

**Рекомендация:** Удалить или заменить на логирование.

---

## High Priority Issues

### 5. Bare `except:` clauses (4 cases)
**Файлы:**
- `src/hipporag/information_extraction/openie_transformers_offline.py:86`
- `src/hipporag/information_extraction/openie_vllm_offline.py:86`
- `src/hipporag/embedding_model/OpenAI.py:111`
- `src/hipporag/search_api/server.py:223`

**Проблема:** Голый `except:` перехватывает все исключения, включая `KeyboardInterrupt`, `SystemExit`, `GeneratorExit`. Это скрывает реальные ошибки и мешает корректному завершению работы.

**Рекомендация:** Заменить на `except Exception:` или конкретные типы исключений.

---

### 6. `print()` вместо логирования (37 cases)
**Файлы:**
- `src/hipporag/rerank.py` (строки 100, 144, 159)
- `src/hipporag/HippoRAG.py` (строка 1317)
- `src/hipporag/prompts/prompt_template_manager.py` (строки 199, 201, 204)

**Проблема:** `print()` выводит в stdout, что не подходит для production-логирования. Нет контроля над уровнем логирования, форматированием, выводом в файлы.

**Рекомендация:** Заменить на `logger.info()`, `logger.error()`, `logger.warning()` в зависимости от контекста.

---

### 7. `raise Exception()` вместо специализированных исключений (2 cases)
**Файлы:**
- `src/hipporag/embedding_model/Transformers.py:44`
- `src/hipporag/embedding_model/Cohere.py:47`

**Проблема:** Общее `Exception` сложно перехватывать и диагностировать.

**Рекомендация:** Использовать специализированные исключения из `exceptions.py` или стандартные (`ValueError`, `RuntimeError`).

---

## Medium Priority Issues

### 8. Отсутствие обработки FileNotFoundError
**Файл:** `src/hipporag/prompts/prompt_template_manager.py:56`

```python
raise FileNotFoundError(f"Templates directory '{self.templates_dir}' does not exist.")
```

**Проблема:** Стандартный `FileNotFoundError` не является частью системы исключений HippoRAG.

**Рекомендация:** Использовать `exceptions.DirectoryNotFoundError` или `exceptions.FileNotFoundError`.

---

### 9. Непоследовательная обработка ошибок в ThreadPoolExecutor
**Файлы:**
- `src/hipporag/HippoRAG.py:11`
- `src/hipporag/StandardRAG.py:11`
- `src/hipporag/information_extraction/openie_openai.py:5`

**Проблема:** ThreadPoolExecutor используется без явной обработки ошибок в futures. Если задача выбросит исключение, оно будет проигнорировано или потеряно.

**Рекомендация:** Добавять обработку `future.exception()` после `as_completed()`.

---

### 10. Multiprocessing Manager без cleanup
**Файл:** `src/hipporag/embedding_model/base.py:221-233`

**Проблема:** `multiprocessing.Manager()` создаётся, но нет явного cleanup при завершении работы.

**Рекомендация:** Добавить метод `close()` или использовать контекстный менеджер.

---

## Low Priority Issues

### 11. Замеченные TODO/FIXME комментарии
Проверить наличие未完成ных задач в коде.

### 12. Отсутствие type hints в некоторых местах
Некоторые функции не имеют аннотаций типов, что усложняет поддержку.

---

## Recommendations

### Immediate Actions (Priority 1)
1. Удалить `assert False` из HippoRAG.py:249
2. Заменить `assert False` на `raise ValueError` в embedding_model/__init__.py:38
3. Удалить `ipdb.set_trace()` из StandardRAG.py:103
4. Удалить `ipdb.set_trace()` из embedding_model/OpenAI.py:114

### Short-term Actions (Priority 2)
5. Заменить все голые `except:` на `except Exception:`
6. Заменить `print()` на логирование в rerank.py и других файлах
7. Заменить `raise Exception()` на специализированные исключения

### Long-term Actions (Priority 3)
8. Внедрить единую систему логирования
9. Добавить обработку ошибок в ThreadPoolExecutor
10. Добавить cleanup для multiprocessing.Manager()

---

## Files to Review
- `src/hipporag/HippoRAG.py`
- `src/hipporag/StandardRAG.py`
- `src/hipporag/embedding_model/__init__.py`
- `src/hipporag/embedding_model/OpenAI.py`
- `src/hipporag/embedding_model/Transformers.py`
- `src/hipporag/embedding_model/Cohere.py`
- `src/hipporag/rerank.py`
- `src/hipporag/information_extraction/openie_transformers_offline.py`
- `src/hipporag/information_extraction/openie_vllm_offline.py`
- `src/hipporag/search_api/server.py`
- `src/hipporag/prompts/prompt_template_manager.py`

---

## Conclusion
Проект требует значительной доработки обработки ошибок и удаления отладочного кода перед использованием в production. Критические проблемы (assert False, ipdb.set_trace()) должны быть исправлены в первую очередь.
