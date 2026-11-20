"""Parameter count and weight memory derived from the architecture.

Weight-only quantization (GPTQ/AWQ style) quantizes the large linear layers
(attention projections and MLP). Embeddings, the LM head and norms are usually
kept in FP16, which ``keep_embeddings_fp16`` models.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import MLP_TYPES, ModelConfig
from .precision import FP16, Precision


@dataclass(frozen=True)
class ParamBreakdown:
    embedding: int
    attention: int  # all layers: Q, K, V, O projections
    mlp: int  # all layers
    norms: int
    lm_head: int  # 0 when tied to the embedding

    @property
    def linear(self) -> int:
        """Parameters in the transformer-block linear layers (what gets quantized)."""
        return self.attention + self.mlp

    @property
    def other(self) -> int:
        return self.embedding + self.norms + self.lm_head

    @property
    def total(self) -> int:
        return self.linear + self.other


def attention_params_per_layer(model: ModelConfig) -> int:
    q_out = model.num_heads * model.head_dim
    kv_out = model.num_kv_heads * model.head_dim
    return model.hidden_size * (q_out + 2 * kv_out) + q_out * model.hidden_size


def mlp_params_per_layer(model: ModelConfig) -> int:
    return MLP_TYPES[model.mlp_type] * model.hidden_size * model.intermediate_size


def param_breakdown(model: ModelConfig) -> ParamBreakdown:
    embedding = model.vocab_size * model.hidden_size
    return ParamBreakdown(
        embedding=embedding,
        attention=model.num_layers * attention_params_per_layer(model),
        mlp=model.num_layers * mlp_params_per_layer(model),
        norms=(2 * model.num_layers + 1) * model.hidden_size,  # two RMSNorms per block + final
        lm_head=0 if model.tie_embeddings else embedding,
    )


def num_params(model: ModelConfig) -> int:
    return param_breakdown(model).total


def weight_bytes(
    model: ModelConfig, weight_precision: Precision = FP16, keep_embeddings_fp16: bool = True
) -> int:
    """Bytes needed to hold the model weights on the GPU."""
    params = param_breakdown(model)
    per_layer_linear = attention_params_per_layer(model) + mlp_params_per_layer(model)
    linear = model.num_layers * weight_precision.bytes_for(per_layer_linear)
    other_precision = FP16 if keep_embeddings_fp16 and weight_precision.bits < 16 else weight_precision
    return linear + other_precision.bytes_for(params.other)
