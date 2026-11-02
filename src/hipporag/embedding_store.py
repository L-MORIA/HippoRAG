import json
import logging
import os
import threading
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)


class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace
        # Мутации хранилища сериализуются: FastAPI-сервер может обращаться параллельно (S-31)
        self._lock = threading.RLock()

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(db_filename, f"vdb_{self.namespace}.parquet")
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {"content": text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Insert strings into the embedding store with optional metadata.

        Args:
            texts: List of text strings to embed and store
            metadatas: Optional list of metadata dictionaries corresponding to each text
        """
        with self._lock:
            self._insert_strings_locked(texts, metadatas)

    def _insert_strings_locked(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        nodes_dict = {}

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {
                "content": text,
                "metadata": metadata,
            }

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        # Одинаковый текст в разных документах (колонтитулы, дисклеймеры) — дописываем
        # источник к метаданным существующей записи вместо тихой потери (S-12).
        metadata_updated = False
        for hash_id in all_hash_ids:
            if hash_id in existing and self._merge_metadata(
                hash_id, nodes_dict[hash_id].get("metadata")
            ):
                metadata_updated = True

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist."
        )

        if metadata_updated:
            logger.info("Added new source metadata entries to existing records.")
            self._save_data()

        if not missing_ids:
            return {}  # All records already exist.

        # Prepare the texts to encode from the "content" field.
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        metadatas_to_save = [nodes_dict[hash_id].get("metadata", {}) for hash_id in missing_ids]

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings, metadatas_to_save)

    @staticmethod
    def _coerce_metadata_entries(meta) -> List[Dict]:
        """Normalize stored metadata into a list of dicts.

        Current format: JSON list of dicts (one entry per source document).
        Backward compatibility: old files keep a single dict or its JSON string.
        """
        if meta is None or meta == "":
            return []
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                return []
        if isinstance(meta, dict):
            return [meta]
        if isinstance(meta, list):
            return [m for m in meta if isinstance(m, dict)]
        return []

    def _merge_metadata(self, hash_id: str, new_meta: Optional[Dict]) -> bool:
        """Append a metadata entry for an existing hash_id; returns True if the record changed."""
        if not new_meta:
            return False
        idx = self.hash_id_to_idx.get(hash_id)
        if idx is None:
            return False
        entries = self.metadatas[idx]
        if any(entry == new_meta for entry in entries):
            return False
        entries.append(new_meta)
        return True

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["content"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()

            # Metadata column may not exist in older versions.
            # Internal storage: list of metadata dicts per hash_id (one entry per source document).
            if "metadata" in df.columns:
                self.metadatas = [
                    self._coerce_metadata_entries(meta) for meta in df["metadata"].values.tolist()
                ]
            else:
                self.metadatas = [[] for _ in self.hash_ids]

            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t, "metadata": m}
                for h, t, m in zip(self.hash_ids, self.texts, self.metadatas)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
            if not (
                len(self.hash_ids) == len(self.texts) == len(self.embeddings) == len(self.metadatas)
            ):
                raise ValueError(
                    f"Повреждённое хранилище {self.filename}: длины колонок не совпадают"
                )
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.metadatas = []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        # Metadata is stored as a JSON list of dicts (one entry per source document)
        metadata_strings = [json.dumps(meta, ensure_ascii=False) for meta in self.metadatas]

        data_to_save = pd.DataFrame(
            {
                "hash_id": self.hash_ids,
                "content": self.texts,
                "embedding": self.embeddings,
                "metadata": metadata_strings,
            }
        )
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t, "metadata": m}
            for h, t, m in zip(self.hash_ids, self.texts, self.metadatas)
        }
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings, metadatas=None):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        # Handle metadata for new entries (internal storage: list of metadata dicts per hash_id)
        if metadatas:
            self.metadatas.extend([[m] if m else [] for m in metadatas])
        else:
            self.metadatas.extend([[] for _ in hash_ids])

        logger.info("Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        with self._lock:
            self._delete_locked(hash_ids)

    def _delete_locked(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)
            self.metadatas.pop(idx)

        logger.info("Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_metadata(self, hash_id):
        """Get the primary (first known) metadata for a specific hash_id."""
        idx = self.hash_id_to_idx[hash_id]
        entries = self.metadatas[idx] if idx < len(self.metadatas) else []
        return entries[0] if entries else {}

    def get_all_metadata_entries(self, hash_id) -> List[Dict]:
        """Get all metadata entries (one per source document) for a specific hash_id."""
        idx = self.hash_id_to_idx.get(hash_id, -1)
        if idx == -1 or idx >= len(self.metadatas):
            return []
        return list(self.metadatas[idx])

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id: self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_metadatas(self, hash_ids):
        """Get metadata for a list of hash_ids."""
        if not hash_ids:
            return []

        metadatas = []
        for hash_id in hash_ids:
            idx = self.hash_id_to_idx.get(hash_id, -1)
            if idx != -1 and idx < len(self.metadatas) and self.metadatas[idx]:
                metadatas.append(self.metadatas[idx][0])
            else:
                metadatas.append({})
        return metadatas

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row["content"] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)

    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings
