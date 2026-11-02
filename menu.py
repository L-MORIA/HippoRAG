#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Меню проверки очищенного индекса HippoRAG.
Запуск: python menu.py
"""

import os
import sys
import subprocess


def clear():
    import subprocess
    subprocess.run('cls' if os.name == 'nt' else 'clear', shell=True)


def print_header(title):
    line = '=' * 82
    print()
    print(line)
    print(f'  {title:^76}')
    print(line)
    print()


def run_script(cmd, description):
    clear()
    print_header(description)
    try:
        subprocess.run(cmd.split(), check=True)
    except subprocess.CalledProcessError:
        print(f'\n[ERROR] Скрипт завершился с ошибкой.')
    except FileNotFoundError:
        print(f'\n[ERROR] Не удалось найти скрипт. Проверьте путь.')
    print()
    input('Нажмите Enter, чтобы вернуться в меню...')


def menu():
    while True:
        clear()
        line = '=' * 82
        print()
        print(line)
        print('  МЕНЮ ПРОВЕРКИ ОЧИЩЕННОГО ИНДЕКСА'.center(78))
        print(line)
        print()
        print('  1. Быстрая проверка индекса (30 сек)')
        print('  2. Интерактивный поиск')
        print('  3. Сравнение результатов поиска (запрос "Овен")')
        print('  4. Статистика индекса')
        print('  5. Очистка индекса от мусорных entities')
        print('  6. Основной поиск (main.py — 7 режимов: от гибридного до HyDE)')
        print()
        print('  0. Выход')
        print()
        print(line)
        print()

        choice = input('Выберите действие (0-6): ').strip()

        if choice == '1':
            run_script('python tests/standalone/check_cleaned_index.py', 'ПРОВЕРКА ОЧИЩЕННОГО ИНДЕКСА')
        elif choice == '2':
            run_script('python tests/standalone/manual_search.py', 'ИНТЕРАКТИВНЫЙ ПОИСК')
        elif choice == '3':
            run_script('python tests/standalone/compare_search_results.py', 'СРАВНЕНИЕ РЕЗУЛЬТАТОВ ПОИСКА')
        elif choice == '4':
            run_script(
                'python scripts/show_index_stats.py --save_dir outputs/astrology_test',
                'СТАТИСТИКА ИНДЕКСА'
            )
        elif choice == '5':
            clear()
            print_header('ОЧИСТКА МУСОРНЫХ ENTITIES')
            print('ВНИМАНИЕ: Эта операция удалит мусорные entities из индекса.')
            print('Backup будет создан автоматически.')
            print()
            confirm = input('Продолжить? (y/n): ').strip().lower()
            if confirm == 'y':
                run_script(
                    'python scripts/clean_garbage_entities.py',
                    'ОЧИСТКА МУСОРНЫХ ENTITIES'
                )
            else:
                print('\nОперация отменена.')
                input('\nНажмите Enter, чтобы вернуться в меню...')
        elif choice == '6':
            clear()
            print_header('ОСНОВНОЙ ПОИСК')
            print('Запуск main.py...')
            print()
            subprocess.run('python main.py', shell=True)
            print()
            input('Нажмите Enter, чтобы вернуться в меню...')
            clear()
            print_header('ОЧИСТКА МУСОРНЫХ ENTITIES')
            print('ВНИМАНИЕ: Эта операция удалит мусорные entities из индекса.')
            print('Backup будет создан автоматически.')
            print()
            confirm = input('Продолжить? (y/n): ').strip().lower()
            if confirm == 'y':
                run_script(
                    'python scripts/clean_garbage_entities.py',
                    'ОЧИСТКА МУСОРНЫХ ENTITIES'
                )
            else:
                print('\nОперация отменена.')
                input('\nНажмите Enter, чтобы вернуться в меню...')
        elif choice == '0':
            clear()
            line = '=' * 82
            print()
            print(line)
            print('  Спасибо за использование!'.center(78))
            print(line)
            print()
            sys.exit(0)
        else:
            print('\nНеверный выбор. Попробуйте снова.')
            input('Нажмите Enter...')


if __name__ == '__main__':
    menu()
