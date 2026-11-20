"""KV-cache size: the core formula of the project.

    KV bytes = 2 x layers x kv_heads x head_dim x context x batch x bytes_per_value

The 2 is one Key and one Value vector. Each (layer, kv_head) vector of
``head_dim`` values is stored and quantized on its own, so quantization groups
never cross head boundaries.

``kv_bytes_per_token`` is the number Parts 2 and 3 build on.
"""

from __future__ import annotations

from .config import ModelConfig
from .precision import FP16, Precision, get_precision


def kv_bytes_per_token_per_layer(model: ModelConfig, kv_precision: Precision = FP16) -> int:
    """Bytes one token adds to the cache of a single layer (K and V, all KV heads)."""
    return 2 * model.num_kv_heads * kv_precision.bytes_for(model.head_dim)


def kv_bytes_per_token(model: ModelConfig, kv_precision: Precision = FP16) -> int:
    """Bytes one token adds to the KV-cache across all layers, for one sequence."""
    return model.num_layers * kv_bytes_per_token_per_layer(model, kv_precision)


def kv_cache_bytes(
    model: ModelConfig, kv_precision: Precision = FP16, context_len: int = 1, batch: int = 1
) -> int:
    """Total KV-cache bytes for ``batch`` sequences that each hold ``context_len`` tokens."""
    if context_len < 0 or batch < 0:
        raise ValueError("context_len and batch must be >= 0")
    return kv_bytes_per_token(model, kv_precision) * context_len * batch


def default_gqa_kv_heads(num_heads: int) -> int:
    """A typical GQA grouping: the largest divisor of num_heads that is <= num_heads/4."""
    for kv in range(num_heads // 4, 1, -1):
        if num_heads % kv == 0:
            return kv
    return 1


def compare_attention(
    model: ModelConfig,
    kv_precision: Precision = FP16,
    context_len: int = 4096,
    batch: int = 1,
    gqa_kv_heads: int | None = None,
) -> list[dict]:
    """KV-cache size of the same model under MHA, GQA and MQA."""
    if gqa_kv_heads is None:
        gqa_kv_heads = (
            model.num_kv_heads if model.attention_type == "GQA"
            else default_gqa_kv_heads(model.num_heads)
        )
    variants = [("MHA", model.num_heads), ("GQA", gqa_kv_heads), ("MQA", 1)]
    mha_total = kv_cache_bytes(model.with_kv_heads(model.num_heads), kv_precision, context_len, batch)
    rows = []
    for label, kv_heads in variants:
        variant = model.with_kv_heads(kv_heads)  # validates divisibility
        total = kv_cache_bytes(variant, kv_precision, context_len, batch)
        rows.append({
            "attention": label,
            "kv_heads": kv_heads,
            "queries_per_kv_head": variant.queries_per_kv_head,
            "bytes_per_token": kv_bytes_per_token(variant, kv_precision),
            "kv_cache_bytes": total,
            "reduction_vs_mha": mha_total / total if total else float("inf"),
            "is_model_default": kv_heads == model.num_kv_heads,
        })
    return rows


def compare_precision(
    model: ModelConfig,
    precisions: list[str | Precision] | None = None,
    context_len: int = 4096,
    batch: int = 1,
) -> list[dict]:
    """KV-cache size of the same model stored in different number formats."""
    chosen = [get_precision(p) for p in (precisions or ["fp32", "fp16", "fp8", "int8", "int4"])]
    fp16_total = kv_cache_bytes(model, FP16, context_len, batch)
    rows = []
    for p in chosen:
        total = kv_cache_bytes(model, p, context_len, batch)
        per_vector = p.bytes_for(model.head_dim)
        rows.append({
            "precision": p.name,
            "bits": p.bits,
            "group_size": p.group_size or "",
            "effective_bits": per_vector * 8 / model.head_dim,
            "bytes_per_token": kv_bytes_per_token(model, p),
            "kv_cache_bytes": total,
            "reduction_vs_fp16": fp16_total / total if total else float("inf"),
            "ideal_reduction_vs_fp16": 16 / p.bits,
        })
    return rows
