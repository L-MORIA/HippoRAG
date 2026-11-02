"""Clean garbage entities from existing index without reindexing."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hipporag.utils.misc_utils import is_valid_entity
import pandas as pd
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_garbage_entities():
    """Remove garbage entities from the existing index."""
    
    # Get working directory - use the most recent index
    working_dir = 'outputs/astrology_test/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2'
    
    entity_path = os.path.join(working_dir, 'entity_embeddings', 'vdb_entity.parquet')
    
    if not os.path.exists(entity_path):
        logger.error(f"Entity embeddings not found at {entity_path}")
        return
    
    logger.info(f"Loading entities from {entity_path}")
    df_entities = pd.read_parquet(entity_path)
    logger.info(f"Total entities in index: {len(df_entities)}")
    
    # Check each entity
    garbage_indices = []
    valid_indices = []
    
    logger.info("Checking entities for validity...")
    for idx, row in df_entities.iterrows():
        entity_name = row['content']
        
        # Check if entity is valid
        if not is_valid_entity(entity_name):
            garbage_indices.append(idx)
            logger.debug(f"Garbage entity: '{entity_name}'")
        else:
            valid_indices.append(idx)
    
    logger.info(f"Found {len(garbage_indices)} garbage entities to remove")
    logger.info(f"Keeping {len(valid_indices)} valid entities")
    
    if garbage_indices:
        # Show some examples
        logger.info("Examples of garbage entities:")
        for idx in garbage_indices[:20]:
            logger.info(f"  - '{df_entities.loc[idx, 'content']}'")
        
        # Remove garbage entities
        logger.info("Removing garbage entities...")
        df_clean = df_entities.drop(garbage_indices)
        
        # Save cleaned data
        backup_path = entity_path + '.backup'
        logger.info(f"Creating backup at {backup_path}")
        df_entities.to_parquet(backup_path)
        
        logger.info(f"Saving cleaned entities to {entity_path}")
        df_clean.to_parquet(entity_path)
        
        logger.info(f"Successfully removed {len(garbage_indices)} garbage entities")
        logger.info(f"Remaining entities: {len(df_clean)}")
    else:
        logger.info("No garbage entities found!")


if __name__ == "__main__":
    clean_garbage_entities()
