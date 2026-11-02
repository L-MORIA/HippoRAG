import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypedDict

from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.llm_utils import filter_invalid_triples, fix_broken_generated_json
from ..utils.logging_utils import get_logger
from ..utils.misc_utils import NerRawOutput, TripleRawOutput, is_valid_entity, normalize_entity_name

MAX_RETRIES = 2  # Total attempts per chunk (including first)
RETRY_JSON_STRICT = ' CRITICAL: Respond with valid JSON only. No markdown, no code fences. Format: {"named_entities": [...]}'
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_list_field(real_response: str, field: str) -> list:
    """Extract a list-typed field from a JSON object response.

    Tries strict json.loads first; falls back to a regex scan when the model wraps
    the JSON in extra text. A bracket inside a string value no longer breaks parsing.
    """
    try:
        data = json.loads(real_response)
    except json.JSONDecodeError:
        pattern = rf'\{{[^{{}}]*"{field}"\s*:\s*\[[^\]]*\][^{{}}]*\}}'
        match = re.search(pattern, real_response, re.DOTALL)
        if match is None:
            return []
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict) and isinstance(data.get(field), list):
        return data[field]
    return []


def _extract_ner_from_response(real_response):
    return _extract_list_field(real_response, "named_entities")


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        all_metadata = {"retry_attempts": 0, "retry_reason": None}
        metadata = {}

        for attempt in range(MAX_RETRIES):
            raw_response = ""
            try:
                ner_input_message = self.prompt_template_manager.render(name="ner", passage=passage)
                if attempt > 0:
                    new_messages = list(ner_input_message)
                    new_messages[0]["content"] += RETRY_JSON_STRICT
                    ner_input_message = new_messages

                raw_response, metadata, cache_hit = self.llm_model.infer(
                    messages=ner_input_message,
                )
                metadata["cache_hit"] = cache_hit

                if metadata["finish_reason"] == "length":
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response

                extracted_entities = _extract_ner_from_response(real_response)
                unique_entities = list(
                    dict.fromkeys(normalize_entity_name(e) for e in extracted_entities)
                )

                # Filter out garbage entities using is_valid_entity
                unique_entities = [e for e in unique_entities if is_valid_entity(e)]

                return NerRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,
                    unique_entities=unique_entities,
                    metadata=metadata,
                )

            except Exception as e:
                logger.warning(f"NER attempt {attempt+1}/{MAX_RETRIES} failed for {chunk_key}: {e}")
                all_metadata["retry_attempts"] = attempt + 1
                all_metadata["retry_reason"] = str(e)
                all_metadata.update(metadata)

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=[],
            metadata=all_metadata,
        )

    def triple_extraction(
        self, chunk_key: str, passage: str, named_entities: List[str]
    ) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            return _extract_list_field(real_response, "triples")

        all_metadata = {"retry_attempts": 0, "retry_reason": None}

        for attempt in range(MAX_RETRIES):
            raw_response = ""
            metadata = {}
            try:
                messages = self.prompt_template_manager.render(
                    name="triple_extraction",
                    passage=passage,
                    named_entity_json=json.dumps({"named_entities": named_entities}),
                )
                if attempt > 0:
                    new_messages = list(messages)
                    new_messages[0]["content"] += RETRY_JSON_STRICT.replace(
                        "named_entities", "triples"
                    ).replace("[...]", "[[...]]")
                    messages = new_messages

                raw_response, metadata, cache_hit = self.llm_model.infer(
                    messages=messages,
                )
                metadata["cache_hit"] = cache_hit

                if metadata["finish_reason"] == "length":
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                extracted_triples = _extract_triples_from_response(real_response)
                triplets = filter_invalid_triples(triples=extracted_triples)

                return TripleRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,
                    metadata=metadata,
                    triples=triplets,
                )

            except Exception as e:
                logger.warning(
                    f"Triple attempt {attempt+1}/{MAX_RETRIES} failed for {chunk_key}: {e}"
                )
                all_metadata["retry_attempts"] = attempt + 1
                all_metadata["retry_reason"] = str(e)
                all_metadata.update(metadata)

        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=all_metadata,
            triples=[],
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(
            chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities
        )
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(
        self, chunks: Dict[str, ChunkInfo]
    ) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        # Concurrent workers for parallel LLM requests
        max_workers = int(os.environ.get("OPENIE_MAX_WORKERS", "15"))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1

                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(
                    self.triple_extraction,
                    ner_result.chunk_id,
                    chunk_passages[ner_result.chunk_id],
                    ner_result.unique_entities,
                ): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1
                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict
