import argparse
import hashlib
import json
import os
import sys

import numpy as np
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def text_fingerprint(text: str) -> str:
    return hashlib.md5(text[:500].encode()).hexdigest()


def mrr_at_k(ranks, k=10):
    reciprocals = [1.0 / r if 0 < r <= k else 0.0 for r in ranks]
    return sum(reciprocals) / len(reciprocals) if ranks else 0.0


def recall_at_k(ranks, k):
    return sum(1 for r in ranks if 0 < r <= k) / len(ranks) if ranks else 0.0


def hit_rate_at_1(ranks):
    return sum(1 for r in ranks if r == 1) / len(ranks) if ranks else 0.0


def evaluate_dpr(hipporag, query, correct_idx):
    query_emb = hipporag.embedding_model.encode([query])
    chunk_store = hipporag.chunk_embedding_store
    embeddings = np.array(chunk_store.embeddings)
    scores = np.dot(embeddings, query_emb.T).squeeze()
    sorted_idx = np.argsort(scores)[::-1]
    try:
        rank = int(np.where(sorted_idx == correct_idx)[0][0]) + 1
    except IndexError:
        rank = 0
    return rank


def evaluate_hipporag(hipporag, query, correct_fingerprint, top_k=10):
    try:
        results = hipporag.retrieve([query], num_to_retrieve=top_k)
    except Exception as e:
        print(f"    HippoRAG retrieve error: {e}")
        return 0
    if not results or not results[0].docs:
        return 0
    for rank, doc_text in enumerate(results[0].docs, 1):
        if text_fingerprint(doc_text) == correct_fingerprint:
            return rank
    return 0


def main():
    parser = argparse.ArgumentParser(description='Evaluate DPR vs HippoRAG retrieval')
    parser.add_argument('--ground-truth', default='eval/ground_truth.jsonl')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--max-queries', type=int, default=0,
                        help='Limit queries (0 = all). Recommended ≤30 for HippoRAG full pipeline')
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        print(f"ERROR: {args.ground_truth} not found. Run generate_ground_truth.py first.")
        sys.exit(1)

    with open(args.ground_truth, 'r', encoding='utf-8') as f:
        ground_truth = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(ground_truth)} ground truth entries")

    eval_pairs = []
    for entry in ground_truth:
        for q in entry['questions']:
            eval_pairs.append({
                'question': q,
                'chunk_id': entry['chunk_id'],
                'fingerprint': entry['chunk_fingerprint'],
            })

    if args.max_queries > 0:
        eval_pairs = eval_pairs[:args.max_queries]
    print(f"Evaluation pairs: {len(eval_pairs)}")

    SAVE_DIR = 'outputs/astrology'
    LLM_NAME = os.getenv('LLM_NAME', 'gemma-4-e4b')
    LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:1234/v1')
    EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    print("Initializing HippoRAG...")
    config = BaseConfig(
        save_dir=SAVE_DIR,
        llm_name=LLM_NAME,
        llm_base_url=LLM_BASE_URL,
        embedding_model_name=EMBEDDING_MODEL,
        linking_top_k=100,
        retrieval_top_k=200,
        passage_node_weight=0.05,
        embedding_batch_size=8,
    )
    hipporag = HippoRAG(global_config=config)

    chunk_store = hipporag.chunk_embedding_store
    chunk_to_idx = {cid: i for i, cid in enumerate(chunk_store.hash_ids)}
    print(f"Chunks in index: {len(chunk_store.hash_ids)}")

    dpr_ranks = []
    hipporag_ranks = []
    skipped = 0

    print("\nEvaluating...")
    for i, pair in enumerate(eval_pairs):
        question = pair['question']
        chunk_id = pair['chunk_id']
        fingerprint = pair['fingerprint']

        if chunk_id not in chunk_to_idx:
            skipped += 1
            continue

        chunk_idx = chunk_to_idx[chunk_id]

        rank = evaluate_dpr(hipporag, question, chunk_idx)
        dpr_ranks.append(rank)

        rank_h = evaluate_hipporag(hipporag, question, fingerprint, args.top_k)
        hipporag_ranks.append(rank_h)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(eval_pairs)}] "
                  f"DPR MRR@10: {mrr_at_k(dpr_ranks, 10):.4f} | "
                  f"HippoRAG MRR@10: {mrr_at_k(hipporag_ranks, 10):.4f}")

    print(f"\nSkipped (chunk not in index): {skipped}")

    print("\n" + "=" * 70)
    print(f"{'Metric':<20} {'DPR':<15} {'HippoRAG':<15}")
    print("=" * 70)
    for k in [1, 5, 10]:
        print(f"{'Recall@' + str(k):<20} "
              f"{recall_at_k(dpr_ranks, k)*100:<15.1f} "
              f"{recall_at_k(hipporag_ranks, k)*100:<15.1f}")
    print(f"{'HitRate@1':<20} "
          f"{hit_rate_at_1(dpr_ranks)*100:<15.1f} "
          f"{hit_rate_at_1(hipporag_ranks)*100:<15.1f}")
    print(f"{'MRR@10':<20} "
          f"{mrr_at_k(dpr_ranks, 10):<15.4f} "
          f"{mrr_at_k(hipporag_ranks, 10):<15.4f}")
    print(f"{'Avg DPR Rank':<20} "
          f"{np.mean([r for r in dpr_ranks if r > 0]):<15.1f} "
          f"{'(N/A for HippoRAG — capped at ' + str(args.top_k) + ')':<15}")
    print("=" * 70)
    print(f"Queries evaluated: {len(dpr_ranks)}")

    results = {
        'total': len(dpr_ranks),
        'skipped': skipped,
        'top_k': args.top_k,
        'dpr': {
            'recall@1': recall_at_k(dpr_ranks, 1),
            'recall@5': recall_at_k(dpr_ranks, 5),
            'recall@10': recall_at_k(dpr_ranks, 10),
            'hitrate@1': hit_rate_at_1(dpr_ranks),
            'mrr@10': mrr_at_k(dpr_ranks, 10),
            'avg_rank': float(np.mean([r for r in dpr_ranks if r > 0])) if any(r > 0 for r in dpr_ranks) else 0,
        },
        'hipporag': {
            'recall@1': recall_at_k(hipporag_ranks, 1),
            'recall@5': recall_at_k(hipporag_ranks, 5),
            'recall@10': recall_at_k(hipporag_ranks, 10),
            'hitrate@1': hit_rate_at_1(hipporag_ranks),
            'mrr@10': mrr_at_k(hipporag_ranks, 10),
        },
    }

    os.makedirs('eval', exist_ok=True)
    with open('eval/metrics_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResults saved to eval/metrics_results.json")


if __name__ == '__main__':
    main()
