from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..prompts.linking import get_query_instruction
from ..utils.config_utils import BaseConfig
from .base import BaseEmbeddingModel


class TransformersEmbeddingModel(BaseEmbeddingModel):
    """
    To select this implementation you can initialise HippoRAG with:
        embedding_model_name starts with "Transformers/"
    """

    def __init__(self, global_config: BaseConfig, embedding_model_name: str) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name.startswith("sentence-transformers/"):
            self.model_id = embedding_model_name[len("sentence-transformers/") :]
        elif embedding_model_name.startswith("Transformers/"):
            self.model_id = embedding_model_name[len("Transformers/") :]
        else:
            self.model_id = embedding_model_name
        self.embedding_type = "float"
        self.batch_size = 64

        self.model = SentenceTransformer(
            self.model_id, device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.search_query_instr = set(
            [get_query_instruction("query_to_fact"), get_query_instruction("query_to_passage")]
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        response = self.model.encode(texts, batch_size=self.batch_size)
        return np.array(response)

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        encode_texts = texts
        if "e5" in self.model_id.lower():
            # Модели intfloat/e5 требуют явных префиксов запроса/документа —
            # без них качество поиска заметно падает
            prefix = "query: " if kwargs.get("instruction") else "passage: "
            encode_texts = [prefix + t for t in texts]

        if len(encode_texts) < self.batch_size:
            return self.encode(encode_texts)

        results = []
        batch_indexes = list(range(0, len(encode_texts), self.batch_size))
        for i in tqdm(batch_indexes, desc="Batch Encoding"):
            results.append(self.encode(encode_texts[i : i + self.batch_size]))
        return np.concatenate(results)
