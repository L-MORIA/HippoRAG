import argparse
import hashlib
import json
import os
import re
import sys
import time

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hipporag.utils.config_utils import BaseConfig
from hipporag.llm import _get_llm_class
from hipporag.llm.llm_utils import TextChatMessage


def text_fingerprint(text: str) -> str:
    return hashlib.md5(text[:500].encode()).hexdigest()


def parse_questions(response: str) -> list[str]:
    text = re.sub(r'^```(?:json)?\s*', '', response.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []
    if isinstance(data, list):
        return [str(q).strip() for q in data if str(q).strip()]
    if isinstance(data, dict):
        for key in ('questions', 'queries', 'results', 'output'):
            if key in data and isinstance(data[key], list):
                return [str(q).strip() for q in data[key] if str(q).strip()]
    return []


def main():
    parser = argparse.ArgumentParser(description='Generate ground truth Q&A from indexed chunks')
    parser.add_argument('--max-chunks', type=int, default=200)
    parser.add_argument('--questions-per-chunk', type=int, default=2)
    parser.add_argument('--output', type=str, default='eval/ground_truth.jsonl')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    SAVE_DIR = 'outputs/astrology'
    LLM_NAME = os.getenv('LLM_NAME', 'gemma-4-e4b')
    LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:1234/v1')
    EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    working_dir = os.path.join(SAVE_DIR, f"{LLM_NAME}_sentence-transformers_paraphrase-multilingual-mpnet-base-v2")
    parquet_path = os.path.join(working_dir, 'chunk_embeddings', 'vdb_chunk.parquet')

    if not os.path.exists(parquet_path):
        print(f"ERROR: Parquet not found: {parquet_path}")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"Total chunks in index: {len(df)}")

    df = df[df['content'].str.len() > 100].reset_index(drop=True)
    print(f"After filtering short chunks: {len(df)}")

    existing_ids = set()
    output_path = args.output
    if args.resume and os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing_ids.add(entry['chunk_id'])
        print(f"Resuming: {len(existing_ids)} chunks already processed")
        df = df[~df['hash_id'].isin(existing_ids)].reset_index(drop=True)
        print(f"Remaining to process: {len(df)}")

    if len(df) > args.max_chunks:
        df = df.sample(args.max_chunks, random_state=42).reset_index(drop=True)
        print(f"Sampled to {args.max_chunks} chunks")

    if len(df) == 0:
        print("Nothing to process.")
        return

    print("Initializing LLM...")
    config = BaseConfig(
        save_dir=SAVE_DIR,
        llm_name=LLM_NAME,
        llm_base_url=LLM_BASE_URL,
        embedding_model_name=EMBEDDING_MODEL,
        response_format={"type": "text"},
    )
    llm = _get_llm_class(config)

    system_prompt = (
        "Ты — эксперт по астрологии и фэншуй. "
        "На основе текста сгенерируй поисковые запросы (вопросы) на русском языке, "
        "на которые отвечает этот текст. "
        "Запросы должны быть конкретными и похожими на то, как их задал бы реальный пользователь. "
        "Отвечай ТОЛЬКО JSON-массивом строк, без пояснений."
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed = 0
    for idx, row in df.iterrows():
        chunk_id = row['hash_id']
        chunk_text = row['content'][:1500]

        user_prompt = (
            f"Текст:\n{chunk_text}\n\n"
            f"Сгенерируй {args.questions_per_chunk} поисковых запроса (вопроса) на русском языке, "
            f"на которые отвечает этот текст. Ответь JSON-массивом строк."
        )

        messages = [
            TextChatMessage(role="system", content=system_prompt),
            TextChatMessage(role="user", content=user_prompt),
        ]

        try:
            response, meta = llm.infer(messages)
            questions = parse_questions(response)
            if not questions:
                print(f"  [{idx+1}/{len(df)}] WARN no questions parsed — {response[:100]}")
                continue

            entry = {
                'chunk_id': chunk_id,
                'chunk_text': row['content'],
                'chunk_fingerprint': text_fingerprint(row['content']),
                'questions': questions,
                'source': row['metadata'].get('source_file', ''),
                'page': row['metadata'].get('page_num', 0),
            }

            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            processed += 1
            print(f"  [{idx+1}/{len(df)}] {chunk_id[:30]}... → {questions}")

        except Exception as e:
            print(f"  [{idx+1}/{len(df)}] ERROR: {e}")
            time.sleep(5)

    print(f"\nDone! Generated questions for {processed} chunks → {output_path}")


if __name__ == '__main__':
    main()
