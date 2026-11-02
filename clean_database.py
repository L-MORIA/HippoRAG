"""
Скрипт для полной очистки всех баз данных HippoRAG.

Удаляет все эмбеддинги, графы, кэши и индексы из указанной директории.
ВНИМАНИЕ: Это необратимая операция!
"""

import shutil
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseCleaner:
    def __init__(self, base_dir: str = 'outputs'):
        """
        Инициализация очистителя баз данных.
        
        Args:
            base_dir: Базовая директория с данными (по умолчанию 'outputs')
        """
        self.base_dir = Path(base_dir)

        if not self.base_dir.exists():
            raise ValueError(f"Директория {base_dir} не существует")

    def _validate_inside_base(self, target: Path) -> Path:
        """Защита от path traversal: цель удаления обязана лежать строго внутри base_dir."""
        base = self.base_dir.resolve()
        resolved = Path(target).resolve()
        if resolved == base or base not in resolved.parents:
            raise ValueError(
                f"Путь удаления '{target}' вне базовой директории '{self.base_dir}' — операция отклонена"
            )
        return resolved

    def list_datasets(self):
        """Показать все датасеты в базовой директории."""
        datasets = [d for d in self.base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        return datasets
    
    def get_dataset_info(self, dataset_dir: Path):
        """Получить информацию о датасете."""
        info = {
            'name': dataset_dir.name,
            'path': str(dataset_dir),
            'size_mb': 0,
            'files': 0,
            'model_dirs': []
        }
        
        # Подсчитать размер и файлы
        for item in dataset_dir.rglob('*'):
            if item.is_file():
                info['files'] += 1
                info['size_mb'] += item.stat().st_size / (1024 * 1024)
        
        # Найти директории моделей
        model_dirs = list(dataset_dir.glob('*_sentence-transformers_*'))
        info['model_dirs'] = [d.name for d in model_dirs]
        
        return info
    
    def clean_dataset(self, dataset_name: str, dry_run: bool = False):
        """
        Очистить конкретный датасет.
        
        Args:
            dataset_name: Имя датасета (например, 'astrology', 'astrology_full')
            dry_run: Если True, только показать что будет удалено
        """
        try:
            dataset_dir = self._validate_inside_base(self.base_dir / dataset_name)
        except ValueError as e:
            logger.error(str(e))
            return False

        if not dataset_dir.exists():
            logger.error(f"Датасет {dataset_name} не найден в {self.base_dir}")
            return False
        
        info = self.get_dataset_info(dataset_dir)
        
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Очистка датасета: {dataset_name}")
        logger.info(f"  Путь: {info['path']}")
        logger.info(f"  Размер: {info['size_mb']:.2f} MB")
        logger.info(f"  Файлов: {info['files']}")
        logger.info(f"  Директорий моделей: {len(info['model_dirs'])}")
        
        if not dry_run:
            try:
                shutil.rmtree(dataset_dir)
                logger.info(f"✓ Датасет {dataset_name} успешно удален")
                return True
            except Exception as e:
                logger.error(f"Ошибка при удалении: {e}")
                return False
        else:
            logger.info(f"[DRY RUN] Будет удалена директория: {dataset_dir}")
            return True
    
    def clean_all(self, dry_run: bool = False, exclude: list = None):
        """
        Очистить все датасеты.
        
        Args:
            dry_run: Если True, только показать что будет удалено
            exclude: Список датасетов для исключения из удаления
        """
        exclude = exclude or []
        datasets = self.list_datasets()
        
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Полная очистка всех датасетов")
        logger.info(f"Найдено датасетов: {len(datasets)}")
        
        total_size = 0
        total_files = 0
        
        for dataset in datasets:
            if dataset.name in exclude:
                logger.info(f"Пропуск {dataset.name} (в списке исключений)")
                continue
            
            info = self.get_dataset_info(dataset)
            total_size += info['size_mb']
            total_files += info['files']
            
            logger.info(f"\n{dataset.name}:")
            logger.info(f"  Размер: {info['size_mb']:.2f} MB")
            logger.info(f"  Файлов: {info['files']}")
        
        logger.info(f"\nИтого:")
        logger.info(f"  Общий размер: {total_size:.2f} MB")
        logger.info(f"  Всего файлов: {total_files}")
        
        if not dry_run:
            for dataset in datasets:
                if dataset.name in exclude:
                    continue
                
                try:
                    shutil.rmtree(self._validate_inside_base(dataset))
                    logger.info(f"✓ Удален: {dataset.name}")
                except Exception as e:
                    logger.error(f"✗ Ошибка при удалении {dataset.name}: {e}")
            
            logger.info("\n✓ Полная очистка завершена!")
        else:
            logger.info("\n[DRY RUN] Для фактического удаления запустите без --dry-run")
    
    def clean_cache_only(self, dataset_name: str = None, dry_run: bool = False):
        """
        Очистить только кэши LLM (не трогая эмбеддинги).
        
        Args:
            dataset_name: Имя датасета или None для всех
            dry_run: Если True, только показать что будет удалено
        """
        if dataset_name:
            try:
                datasets = [self._validate_inside_base(self.base_dir / dataset_name)]
            except ValueError as e:
                logger.error(str(e))
                return
        else:
            datasets = self.list_datasets()
        
        total_size = 0
        total_files = 0
        
        for dataset in datasets:
            if not dataset.exists():
                continue
            
            cache_dir = dataset / 'llm_cache'
            if cache_dir.exists():
                # Подсчитать размер кэша
                cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                cache_files = sum(1 for f in cache_dir.rglob('*') if f.is_file())
                
                total_size += cache_size
                total_files += cache_files
                
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}Кэш в {dataset.name}:")
                logger.info(f"  Размер: {cache_size:.2f} MB")
                logger.info(f"  Файлов: {cache_files}")
                
                if not dry_run:
                    try:
                        shutil.rmtree(cache_dir)
                        logger.info(f"✓ Кэш удален")
                    except Exception as e:
                        logger.error(f"✗ Ошибка: {e}")
        
        logger.info(f"\nИтого кэшей:")
        logger.info(f"  Размер: {total_size:.2f} MB")
        logger.info(f"  Файлов: {total_files}")


def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python clean_database.py --list")
        print("  python clean_database.py --clean-dataset <name> [--dry-run]")
        print("  python clean_database.py --clean-all [--dry-run] [--exclude dataset1,dataset2]")
        print("  python clean_database.py --clean-cache [--dataset <name>] [--dry-run]")
        print("\nПримеры:")
        print("  python clean_database.py --list")
        print("  python clean_database.py --clean-dataset astrology --dry-run")
        print("  python clean_database.py --clean-dataset astrology")
        print("  python clean_database.py --clean-all --dry-run")
        print("  python clean_database.py --clean-all --exclude astrology_full")
        print("  python clean_database.py --clean-cache --dry-run")
        sys.exit(1)
    
    cleaner = DatabaseCleaner('outputs')
    
    if '--list' in sys.argv:
        # Показать список датасетов
        datasets = cleaner.list_datasets()
        print("\nДатасеты в базе данных:")
        print("=" * 80)
        
        total_size = 0
        for dataset in datasets:
            info = cleaner.get_dataset_info(dataset)
            print(f"\n{info['name']}:")
            print(f"  Путь: {info['path']}")
            print(f"  Размер: {info['size_mb']:.2f} MB")
            print(f"  Файлов: {info['files']}")
            print(f"  Модели: {', '.join(info['model_dirs']) if info['model_dirs'] else 'нет'}")
            total_size += info['size_mb']
        
        print("\n" + "=" * 80)
        print(f"Всего датасетов: {len(datasets)}")
        print(f"Общий размер: {total_size:.2f} MB")
    
    elif '--clean-dataset' in sys.argv:
        # Очистить конкретный датасет
        idx = sys.argv.index('--clean-dataset')
        if idx + 1 >= len(sys.argv):
            print("Ошибка: укажите имя датасета после --clean-dataset")
            sys.exit(1)
        
        dataset_name = sys.argv[idx + 1]
        dry_run = '--dry-run' in sys.argv
        
        if not dry_run:
            if not sys.stdin.isatty():
                print("Ошибка: интерактивное подтверждение недоступно (stdin не TTY). Используйте --dry-run.")
                sys.exit(1)
            response = input(f"\n⚠️  ВНИМАНИЕ: Вы уверены, что хотите удалить датасет '{dataset_name}'?\n"
                           f"Это действие необратимо! (yes/no): ")
            if response.lower() != 'yes':
                print("Отменено.")
                sys.exit(0)
        
        cleaner.clean_dataset(dataset_name, dry_run=dry_run)
    
    elif '--clean-all' in sys.argv:
        # Очистить все датасеты
        dry_run = '--dry-run' in sys.argv
        exclude = []
        
        if '--exclude' in sys.argv:
            idx = sys.argv.index('--exclude')
            if idx + 1 < len(sys.argv):
                exclude = sys.argv[idx + 1].split(',')
        
        if not dry_run:
            if not sys.stdin.isatty():
                print("Ошибка: интерактивное подтверждение недоступно (stdin не TTY). Используйте --dry-run.")
                sys.exit(1)
            response = input(f"\n⚠️  ВНИМАНИЕ: Вы уверены, что хотите удалить ВСЕ датасеты?\n"
                           f"Это действие необратимо! (yes/no): ")
            if response.lower() != 'yes':
                print("Отменено.")
                sys.exit(0)
        
        cleaner.clean_all(dry_run=dry_run, exclude=exclude)
    
    elif '--clean-cache' in sys.argv:
        # Очистить только кэши
        dry_run = '--dry-run' in sys.argv
        dataset_name = None
        
        if '--dataset' in sys.argv:
            idx = sys.argv.index('--dataset')
            if idx + 1 < len(sys.argv):
                dataset_name = sys.argv[idx + 1]
        
        if not dry_run:
            if not sys.stdin.isatty():
                print("Ошибка: интерактивное подтверждение недоступно (stdin не TTY). Используйте --dry-run.")
                sys.exit(1)
            response = input(f"\nВы уверены, что хотите очистить кэши? (yes/no): ")
            if response.lower() != 'yes':
                print("Отменено.")
                sys.exit(0)
        
        cleaner.clean_cache_only(dataset_name, dry_run=dry_run)
    
    else:
        print("Ошибка: укажите --list, --clean-dataset, --clean-all или --clean-cache")
        sys.exit(1)


if __name__ == '__main__':
    main()
