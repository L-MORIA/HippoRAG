"""
Мониторинг процесса индексации HippoRAG.
Проверяет прогресс каждые 5 минут и уведомляет о завершении.
"""

import time
import pandas as pd
from pathlib import Path
import subprocess
import sys
from datetime import datetime

PARQUET_FILE = 'outputs/astrology_full/gemma-4-e4b_sentence-transformers_all-MiniLM-L6-v2/chunk_embeddings/vdb_chunk.parquet'
TOTAL_CHUNKS = 1784
LOG_FILE = 'indexing_full.log'
CHECK_INTERVAL = 300  # 5 минут

def get_progress():
    """Получить текущий прогресс индексации."""
    try:
        if Path(PARQUET_FILE).exists():
            df = pd.read_parquet(PARQUET_FILE)
            return len(df)
        return 0
    except Exception as e:
        print(f"Ошибка при чтении прогресса: {e}")
        return 0

def is_process_running():
    """Проверить, запущен ли процесс индексации."""
    try:
        result = subprocess.run(
            ['powershell', '-Command', 'Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.CPU -gt 10 }'],
            capture_output=True,
            text=True
        )
        return 'python' in result.stdout.lower()
    except:
        return False

def get_last_log_line():
    """Получить последнюю строку из лога."""
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Найти последнюю строку с прогрессом
            for line in reversed(lines):
                if 'NER:' in line or 'Triple:' in line or 'Batch' in line:
                    return line.strip()
        return "Нет данных"
    except:
        return "Лог недоступен"

def main():
    print("=" * 80)
    print("МОНИТОРИНГ ИНДЕКСАЦИИ HIPPORAG")
    print("=" * 80)
    print(f"Начало мониторинга: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Всего чанков для обработки: {TOTAL_CHUNKS}")
    print(f"Интервал проверки: {CHECK_INTERVAL // 60} минут")
    print("=" * 80)
    print()
    
    last_progress = 0
    start_time = time.time()
    
    while True:
        current_progress = get_progress()
        percentage = (current_progress / TOTAL_CHUNKS) * 100
        is_running = is_process_running()
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        elapsed = time.time() - start_time
        elapsed_hours = elapsed / 3600
        
        print(f"[{timestamp}] Прогресс: {current_progress}/{TOTAL_CHUNKS} ({percentage:.1f}%)")
        
        if current_progress > last_progress:
            chunks_processed = current_progress - last_progress
            speed = chunks_processed / (CHECK_INTERVAL / 60)  # чанков в минуту
            remaining = TOTAL_CHUNKS - current_progress
            eta_minutes = remaining / speed if speed > 0 else 0
            
            print(f"           Обработано за последние {CHECK_INTERVAL // 60} мин: {chunks_processed} чанков")
            print(f"           Скорость: {speed:.1f} чанков/мин")
            print(f"           Осталось: ~{eta_minutes:.0f} минут ({eta_minutes/60:.1f} часов)")
        
        print(f"           Процесс: {'✓ Работает' if is_running else '✗ Не найден'}")
        
        last_log = get_last_log_line()
        if 'NER:' in last_log or 'Triple:' in last_log:
            # Извлечь процент из строки прогресса
            if '%|' in last_log:
                try:
                    percent_str = last_log.split('%|')[0].split()[-1]
                    print(f"           Текущий батч: {percent_str}%")
                except:
                    pass
        
        print()
        
        # Проверка завершения
        if current_progress >= TOTAL_CHUNKS:
            print("=" * 80)
            print("🎉 ИНДЕКСАЦИЯ ЗАВЕРШЕНА!")
            print("=" * 80)
            print(f"Время выполнения: {elapsed_hours:.2f} часов")
            print(f"Обработано чанков: {current_progress}")
            print()
            print("Теперь можно тестировать систему:")
            print("  python search_test.py")
            print()
            break
        
        if not is_running and current_progress < TOTAL_CHUNKS:
            print("⚠️  ВНИМАНИЕ: Процесс индексации не найден, но индексация не завершена!")
            print(f"   Обработано: {current_progress}/{TOTAL_CHUNKS}")
            print("   Возможно, процесс завершился с ошибкой.")
            print()
            response = input("Продолжить мониторинг? (y/n): ")
            if response.lower() != 'y':
                break
        
        last_progress = current_progress
        
        # Ждем следующей проверки
        print(f"Следующая проверка через {CHECK_INTERVAL // 60} минут...")
        print("-" * 80)
        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nМониторинг остановлен пользователем.")
        sys.exit(0)
