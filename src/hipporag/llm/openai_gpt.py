import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import AzureOpenAI, OpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = (
            getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        )
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute(
                "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                (key_hash, message, metadata_str),
            )
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper


def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)

    return wrapper


class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""

    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        config_dict["max_retries"] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(
        self,
        cache_dir,
        global_config,
        cache_filename: str = None,
        high_throughput: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = (
            getattr(global_config, "llm_api_name", None)
            or os.environ.get("LLM_NAME")
            or global_config.llm_name
        )
        # Check environment for LM Studio or other local LLM
        self.llm_base_url = os.environ.get("LLM_BASE_URL", global_config.llm_base_url)

        logger.info(f"LLM Config: name={self.llm_name}, base_url={self.llm_base_url}")

        # If using local LLM (LM Studio), set dummy API key
        if self.llm_base_url and self.llm_base_url != "None" and self.llm_base_url.strip():
            os.environ["OPENAI_API_KEY"] = "sk-dummy"
            logger.info(f"Using LOCAL LLM at {self.llm_base_url}")

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(limits=limits, timeout=httpx.Timeout(5 * 60, read=5 * 60))
        else:
            client = None

        self.max_retries = kwargs.get("max_retries", 2)

        # Check if using local LLM
        _base_url = self.llm_base_url
        is_local = _base_url and _base_url.strip() and _base_url != "None"

        # For local LLM, use more retries
        if is_local:
            self.max_retries = 5
            logger.info(f"Local LLM detected at {_base_url}, using max_retries={self.max_retries}")

        # Only create OpenAI clients for non-local models (OpenAI API or Azure)
        if self.global_config.azure_endpoint is None and self.global_config.llm_base_url is None:
            # Standard OpenAI API
            self.openai_client = OpenAI(http_client=client, max_retries=self.max_retries)
        elif self.global_config.azure_endpoint is not None:
            # Azure OpenAI
            self.openai_client = AzureOpenAI(
                api_version=self.global_config.azure_endpoint.split("api-version=")[1],
                azure_endpoint=self.global_config.azure_endpoint,
                max_retries=self.max_retries,
            )
        # For local LLMs, we use direct requests in the infer method

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict["llm_name"] = self.global_config.llm_name
        config_dict["llm_base_url"] = self.global_config.llm_base_url
        config_dict["generate_params"] = {
            "model": self.global_config.llm_name,
            "max_completion_tokens": config_dict.get("max_new_tokens", 400),
            "n": config_dict.get("num_gen_choices", 1),
            "seed": config_dict.get("seed", 0),
            "temperature": config_dict.get("temperature", 0.0),
        }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @dynamic_retry_decorator
    @cache_response
    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[str, dict, bool]:
        """Возвращает (content, metadata, cache_hit) — единый контракт для всех потребителей."""
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages

        # For LOCAL LLM, use requests directly instead of OpenAI library
        if self.llm_base_url and self.llm_base_url.strip():
            return self._infer_local_llm(params)

        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if "gpt" not in params["model"] or version.parse(openai.__version__) < version.parse(
            "1.45.0"
        ):  # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params["max_tokens"] = params.pop("max_completion_tokens")

        response = self.openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"

        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata

    def _infer_local_llm(self, params):
        """Call local LLM (LM Studio, vLLM) directly via requests."""
        import json

        import requests

        # Build the request
        url = f"{self.llm_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}

        # Prepare payload - use max_tokens (not max_completion_tokens) for local LLM
        payload = {
            "model": params.get("model") or self.llm_name,
            "messages": params["messages"],
            "max_tokens": params.get("max_completion_tokens", 400),
            "temperature": params.get("temperature", self.global_config.temperature),
            "top_p": params.get("top_p"),
            "n": params.get("n", 1),
        }

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        logger.info(f"Local LLM request to {url}")
        logger.info(f"Model: {payload['model']}")
        logger.info(f"Max tokens: {payload.get('max_tokens')}")
        logger.info(f"Temperature: {payload.get('temperature')}")
        logger.info(
            f"Messages length: {len(params['messages'])} messages, total chars: {sum(len(str(m)) for m in params['messages'])}"
        )
        logger.debug(f"Full payload:\n{json.dumps(payload, ensure_ascii=False)[:1000]}")

        try:
            # Increase timeout for local LLM to 300 seconds (5 minutes)
            resp = requests.post(url, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()

            # Extract response - handle different formats from different models
            message = result["choices"][0]["message"]
            # Handle both string content and dict content
            if isinstance(message, dict):
                response_message = message.get("content", "")
                # Some models use reasoning_content
                if not response_message:
                    response_message = message.get("reasoning_content", "")
            else:
                response_message = str(message)

            usage = result.get("usage", {})

            metadata = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "finish_reason": result["choices"][0].get("finish_reason", "stop"),
            }

            logger.info(f"Local LLM response length: {len(response_message)} chars")
            logger.info(f"First 200 chars of response:\n{str(response_message)[:200]}")

            return response_message, metadata

        except Exception as e:
            logger.error(f"Local LLM request failed: {type(e).__name__}: {e}")
            if "resp" in locals():
                logger.error(f"Response status: {resp.status_code}, body: {resp.text[:500]}")
            raise
