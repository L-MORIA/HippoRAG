import os

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseLLM
from .bedrock_llm import BedrockLLM
from .bedrock_mantle import BedrockMantleLLM
from .openai_gpt import CacheOpenAI
from .transformers_llm import TransformersLLM

logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    # Для локальных серверов (LM Studio, vLLM) не требуется API ключ
    if config.llm_base_url is not None:
        # LM Studio или vLLM на localhost
        if "localhost" in config.llm_base_url or "127.0.0.1" in config.llm_base_url:
            os.environ["OPENAI_API_KEY"] = "sk-dummy"  # dummy key для локального сервера

    if config.llm_name.startswith(BedrockMantleLLM.prefix):
        return BedrockMantleLLM(config)

    if config.llm_name.startswith("bedrock"):
        return BedrockLLM(config)

    if config.llm_name.startswith("Transformers/"):
        return TransformersLLM(config)

    return CacheOpenAI.from_experiment_config(config)


__all__ = [
    "CacheOpenAI",
    "BaseLLM",
    "BedrockLLM",
    "BedrockMantleLLM",
    "TransformersLLM",
    "_get_llm_class",
]
