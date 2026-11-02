from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig
from .Cohere import CohereEmbeddingModel
from .Contriever import ContrieverModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from .Transformers import TransformersEmbeddingModel
from .VLLM import VLLMEmbeddingModel

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    elif "cohere" in embedding_model_name:
        return CohereEmbeddingModel
    elif embedding_model_name.startswith("Transformers/"):
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("VLLM/"):
        return VLLMEmbeddingModel
    elif embedding_model_name.startswith("sentence-transformers/"):
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("intfloat/"):
        return TransformersEmbeddingModel
    elif "MiniLM" in embedding_model_name or "all-MiniLM" in embedding_model_name:
        return TransformersEmbeddingModel
    elif "multilingual" in embedding_model_name.lower():
        return TransformersEmbeddingModel
    raise ValueError(f"Unknown embedding model name: {embedding_model_name}")


__all__ = [
    "ContrieverModel",
    "EmbeddingConfig",
    "BaseEmbeddingModel",
    "GritLMEmbeddingModel",
    "NVEmbedV2EmbeddingModel",
    "OpenAIEmbeddingModel",
    "CohereEmbeddingModel",
    "TransformersEmbeddingModel",
    "VLLMEmbeddingModel",
    "_get_embedding_model_class",
]
