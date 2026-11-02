"""Реранкеры: cross-encoder (en/ru), E5, ColBERT-v2 — ленивые синглтоны.

Перенос из main.py (шаг 5 миграции MAIN_SPLIT_PLAN.md) — тела извлечены
байт-в-байт (849-919, 922, 1012-1020, 1524-1589); глобальные синглтоны
(_CROSS_ENCODER, _E5_MODEL, _COLBERT*) переезжают вместе с функциями.
"""

import numpy as np

from hipporag.utils.misc_utils import min_max_normalize

_CROSS_ENCODER = None
_CROSS_ENCODER_RU = None


def _get_crossencoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER


def _get_crossencoder_ru():
    global _CROSS_ENCODER_RU
    if _CROSS_ENCODER_RU is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER_RU = CrossEncoder("DiTy/cross-encoder-russian-msmarco")
    return _CROSS_ENCODER_RU


def _rerank_with_crossencoder_ru(query, candidates):
    if not candidates:
        return candidates
    try:
        ce = _get_crossencoder_ru()
        pairs = [(query, c[0][:512]) for c in candidates]
        xc_scores = ce.predict(pairs, show_progress_bar=False)
        # Сырой CE-скор у RU-модели — вероятность (0…1), у EN — логит (…±11),
        # а old — embedding-скор (0…1). Смешивать их напрямую бессмысленно:
        # коэффициент 0.7/0.3 не означает ничего. Нормируем xc (P1-4).
        xc_scores = min_max_normalize(np.array([float(x) for x in xc_scores]))
        reranked = []
        for i, c in enumerate(candidates):
            xc = float(xc_scores[i])
            text = c[0]
            old = c[1]
            boost = c[4] if len(c) > 4 else 0.0
            final = xc * 0.7 + old * 0.3 * (1.0 + boost)
            reranked.append((text, final, xc, old, boost))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    except Exception as e:
        print(f"  \u26a0\ufe0f Русский Cross-encoder недоступен ({e})")
        return candidates


def _rerank_with_crossencoder(query, candidates):
    if not candidates:
        return candidates
    try:
        ce = _get_crossencoder()
        pairs = [(query, c[0][:512]) for c in candidates]
        xc_scores = ce.predict(pairs, show_progress_bar=False)
        # Нормировка CE-скора перед смешиванием — см. P1-4 (в EN-ветке сырой
        # логит давал 86 % веса в итоговом скоре и переставлял кандидатов).
        xc_scores = min_max_normalize(np.array([float(x) for x in xc_scores]))
        reranked = []
        for i, c in enumerate(candidates):
            xc = float(xc_scores[i])
            text = c[0]
            old = c[1]
            boost = c[4] if len(c) > 4 else 0.0
            final = xc * 0.7 + old * 0.3 * (1.0 + boost)
            reranked.append((text, final, xc, old, boost))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    except Exception as e:
        print(f"  \u26a0\ufe0f Cross-encoder недоступен ({e}), используется исходный порядок")
        return candidates


_E5_MODEL = None


def _get_e5():
    global _E5_MODEL
    if _E5_MODEL is None:
        from sentence_transformers import SentenceTransformer

        print("  ⏳ Загрузка E5 модели...", flush=True)
        _E5_MODEL = SentenceTransformer("intfloat/multilingual-e5-small")
        print("  ✅ E5 модель загружена", flush=True)
    return _E5_MODEL


_COLBERT = None
_COLBERT_LINEAR = None
_COLBERT_BERT = None


def _get_colbert():
    """Загрузить ColBERT-v2 (лениво): BERT + linear projection (128-dim)."""
    global _COLBERT, _COLBERT_LINEAR, _COLBERT_BERT
    if _COLBERT is None:
        import os as _os

        from safetensors.torch import load_file
        from sentence_transformers import SentenceTransformer

        print("  ⏳ Загрузка ColBERT-v2...", flush=True)
        st = SentenceTransformer("colbert-ir/colbertv2.0")
        _COLBERT_BERT = st[0].auto_model
        _COLBERT_BERT.eval()
        hub = _os.path.expanduser(
            "~/.cache/huggingface/hub/models--colbert-ir--colbertv2.0/snapshots"
        )
        if _os.path.exists(hub):
            snaps = _os.listdir(hub)
            if snaps:
                safetensors = load_file(_os.path.join(hub, snaps[0], "model.safetensors"))
                _COLBERT_LINEAR = safetensors["linear.weight"]  # (128, 768)
        _COLBERT = st
        print("  ✅ ColBERT-v2 загружен", flush=True)
    return _COLBERT


def _colbert_encode(text, max_length=512):
    """Encode text with ColBERT → 128-dim token embeddings + mask."""
    import torch
    import torch.nn.functional as F

    tokens = _COLBERT.tokenize([text])
    for k, v in tokens.items():
        if isinstance(v, torch.Tensor) and v.shape[1] > max_length:
            tokens[k] = v[:, :max_length]
    # Токенизатор возвращает тензоры на CPU, а модель может быть на CUDA:
    # token_type_ids в слое эмбеддингов идёт через index_select и падает с
    # "Expected all tensors to be on the same device" (P2-7).
    device = next(_COLBERT_BERT.parameters()).device
    tokens = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in tokens.items()}
    with torch.no_grad():
        out = _COLBERT_BERT(**tokens)
        emb = out.last_hidden_state
        if _COLBERT_LINEAR is not None:
            emb = torch.matmul(emb, _COLBERT_LINEAR.T.to(emb.device))
        emb = F.normalize(emb, dim=-1)
    return emb, tokens["attention_mask"]


def _colbert_score(q_emb, q_mask, d_emb, d_mask):
    """MaxSim late interaction score."""
    import torch

    sim = torch.matmul(q_emb, d_emb.transpose(1, 2))
    qm = q_mask.unsqueeze(-1).float()
    dm = d_mask.unsqueeze(1).float()
    sim = sim * qm * dm
    max_sim = sim.max(dim=-1).values
    score = (max_sim * qm.squeeze(-1)).sum(dim=-1)
    score = score / qm.squeeze(-1).sum(dim=-1).clamp(min=1)
    return score.item()
