"""
Скрипт для удаления конкретного документа из HippoRAG по имени файла или source_file.

Удаляет все чанки, entity embeddings и fact embeddings, связанные с документом.
"""

import pandas as pd
import json
import sys
from pathlib import Path
from typing import List, Dict, Set
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentDeleter:
    def __init__(self, output_dir: str):
        """
        Инициализация удалятора документов.
        
        Args:
            output_dir: Путь к директории с данными (например, 'outputs/astrology')
        """
        self.output_dir = Path(output_dir)
        self.model_dirs = list(self.output_dir.glob('*_sentence-transformers_*'))
        
        if not self.model_dirs:
            raise ValueError(f"Не найдены директории моделей в {output_dir}")
        
        logger.info(f"Найдено {len(self.model_dirs)} директорий моделей")
    
    def list_documents(self) -> Dict[str, int]:
        """Список всех документов и количество чанков."""
        documents = {}
        
        for model_dir in self.model_dirs:
            chunk_file = model_dir / 'chunk_embeddings' / 'vdb_chunk.parquet'
            
            if not chunk_file.exists():
                continue
            
            df = pd.read_parquet(chunk_file)
            
            for _, row in df.iterrows():
                meta = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                source = meta.get('source_file', 'unknown')
                documents[source] = documents.get(source, 0) + 1
        
        return documents
    
    def find_chunks_by_source(self, source_file: str, parquet_file: Path) -> List[str]:
        """Найти все hash_id чанков для конкретного документа."""
        if not parquet_file.exists():
            return []
        
        df = pd.read_parquet(parquet_file)
        chunk_hashes = []
        
        for _, row in df.iterrows():
            meta = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
            if meta.get('source_file', '') == source_file:
                chunk_hashes.append(row['hash_id'])
        
        return chunk_hashes
    
    def delete_by_hash_ids(self, parquet_file: Path, hash_ids: Set[str]) -> int:
        """Удалить записи по hash_id из parquet файла."""
        if not parquet_file.exists():
            return 0
        
        df = pd.read_parquet(parquet_file)
        original_count = len(df)
        
        # Фильтруем записи
        df_filtered = df[~df['hash_id'].isin(hash_ids)]
        deleted_count = original_count - len(df_filtered)
        
        if deleted_count > 0:
            # Сохраняем обновленный файл
            df_filtered.to_parquet(parquet_file, index=False)
            logger.info(f"Удалено {deleted_count} записей из {parquet_file.name}")
        
        return deleted_count
    
    def delete_document(self, source_file: str, dry_run: bool = False) -> Dict[str, int]:
        """
        Удалить документ и все связанные данные.
        
        Args:
            source_file: Имя файла источника (например, 'Building a High.pdf')
            dry_run: Если True, только показать что будет удалено без фактического удаления
        
        Returns:
            Словарь с количеством удаленных записей по типам
        """
        stats = {
            'chunks': 0,
            'entities': 0,
            'facts': 0
        }
        
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Удаление документа: {source_file}")
        
        for model_dir in self.model_dirs:
            logger.info(f"Обработка директории: {model_dir.name}")
            
            # 1. Найти все chunk hash_ids для этого документа
            chunk_file = model_dir / 'chunk_embeddings' / 'vdb_chunk.parquet'
            chunk_hashes = self.find_chunks_by_source(source_file, chunk_file)
            
            if not chunk_hashes:
                logger.info(f"  Чанки не найдены в {model_dir.name}")
                continue
            
            logger.info(f"  Найдено {len(chunk_hashes)} чанков для удаления")
            chunk_hash_set = set(chunk_hashes)
            
            if not dry_run:
                # 2. Удалить chunk embeddings
                deleted = self.delete_by_hash_ids(chunk_file, chunk_hash_set)
                stats['chunks'] += deleted
                
                # 3. Удалить entity embeddings (связанные с чанками)
                entity_file = model_dir / 'entity_embeddings' / 'vdb_entity.parquet'
                if entity_file.exists():
                    # Entities могут быть связаны через metadata или content
                    # Для простоты удаляем entities, которые упоминают chunk hash_ids в metadata
                    deleted = self.delete_entities_by_chunks(entity_file, chunk_hash_set)
                    stats['entities'] += deleted
                
                # 4. Удалить fact embeddings (связанные с чанками)
                fact_file = model_dir / 'fact_embeddings' / 'vdb_fact.parquet'
                if fact_file.exists():
                    deleted = self.delete_facts_by_chunks(fact_file, chunk_hash_set)
                    stats['facts'] += deleted
            else:
                logger.info(f"  [DRY RUN] Будет удалено {len(chunk_hashes)} чанков")
                stats['chunks'] += len(chunk_hashes)
        
        logger.info(f"\n{'[DRY RUN] ' if dry_run else ''}Итого удалено:")
        logger.info(f"  Чанков: {stats['chunks']}")
        logger.info(f"  Entities: {stats['entities']}")
        logger.info(f"  Facts: {stats['facts']}")
        
        return stats
    
    def delete_entities_by_chunks(self, entity_file: Path, chunk_hashes: Set[str]) -> int:
        """Удалить entities, связанные с удаленными чанками."""
        if not entity_file.exists():
            return 0
        
        df = pd.read_parquet(entity_file)
        original_count = len(df)
        
        # Фильтруем entities, которые связаны с удаленными чанками
        # Проверяем metadata на наличие ссылок на chunk_hashes
        to_delete = set()
        for idx, row in df.iterrows():
            meta = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
            # Проверяем различные поля metadata
            chunk_refs = meta.get('chunk_ids', []) or meta.get('source_chunks', [])
            if isinstance(chunk_refs, str):
                chunk_refs = [chunk_refs]
            
            if any(chunk_id in chunk_hashes for chunk_id in chunk_refs):
                to_delete.add(row['hash_id'])
        
        if to_delete:
            df_filtered = df[~df['hash_id'].isin(to_delete)]
            deleted_count = original_count - len(df_filtered)
            df_filtered.to_parquet(entity_file, index=False)
            logger.info(f"Удалено {deleted_count} entities из {entity_file.name}")
            return deleted_count
        
        return 0
    
    def delete_facts_by_chunks(self, fact_file: Path, chunk_hashes: Set[str]) -> int:
        """Удалить facts, связанные с удаленными чанками."""
        if not fact_file.exists():
            return 0
        
        df = pd.read_parquet(fact_file)
        original_count = len(df)
        
        # Аналогично entities
        to_delete = set()
        for idx, row in df.iterrows():
            meta = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
            chunk_refs = meta.get('chunk_ids', []) or meta.get('source_chunks', [])
            if isinstance(chunk_refs, str):
                chunk_refs = [chunk_refs]
            
            if any(chunk_id in chunk_hashes for chunk_id in chunk_refs):
                to_delete.add(row['hash_id'])
        
        if to_delete:
            df_filtered = df[~df['hash_id'].isin(to_delete)]
            deleted_count = original_count - len(df_filtered)
            df_filtered.to_parquet(fact_file, index=False)
            logger.info(f"Удалено {deleted_count} facts из {fact_file.name}")
            return deleted_count
        
        return 0


def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python delete_document.py <output_dir> [--list]")
        print("  python delete_document.py <output_dir> --delete <source_file> [--dry-run]")
        print("\nПримеры:")
        print("  python delete_document.py outputs/astrology --list")
        print("  python delete_document.py outputs/astrology --delete 'Building a High.pdf' --dry-run")
        print("  python delete_document.py outputs/astrology --delete 'Building a High.pdf'")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    deleter = DocumentDeleter(output_dir)
    
    if '--list' in sys.argv:
        # Показать список документов
        docs = deleter.list_documents()
        print("\nДокументы в базе данных:")
        print("-" * 80)
        for source, count in sorted(docs.items(), key=lambda x: x[1], reverse=True):
            print(f"{source}: {count} чанков")
        print("-" * 80)
        print(f"Всего документов: {len(docs)}")
    
    elif '--delete' in sys.argv:
        # Удалить документ
        delete_idx = sys.argv.index('--delete')
        if delete_idx + 1 >= len(sys.argv):
            print("Ошибка: укажите имя файла после --delete")
            sys.exit(1)
        
        source_file = sys.argv[delete_idx + 1]
        dry_run = '--dry-run' in sys.argv
        
        # Подтверждение
        if not dry_run:
            response = input(f"\nВы уверены, что хотите удалить '{source_file}'? (yes/no): ")
            if response.lower() != 'yes':
                print("Отменено.")
                sys.exit(0)
        
        stats = deleter.delete_document(source_file, dry_run=dry_run)
        
        if not dry_run:
            print("\n✓ Документ успешно удален!")
    else:
        print("Ошибка: укажите --list или --delete")
        sys.exit(1)


if __name__ == '__main__':
    main()
