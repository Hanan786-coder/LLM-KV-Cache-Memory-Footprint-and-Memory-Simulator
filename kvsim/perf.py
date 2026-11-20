"""Bytes moved, FLOPs and roofline latency of one forward pass.

One forward pass (a prefill or a decode step) has to:

* stream the model weights from GPU memory once (shared by every token in the batch),
* read the KV-cache that attention looks at and write the new KV entries,
* do about 2 FLOPs per weight per token, plus the attention FLOPs.

The roofline model turns that into time:

    t_memory  = HBM bytes / (bandwidth x bw_efficiency)
    t_compute = FLOPs     / (peak FP16 FLOP/s x compute_efficiency)
    latency   = max(t_memory, t_compute) + t_swap + step_overhead

The step is memory-bound when t_memory is the larger term, i.e. when its
arithmetic intensity (FLOPs / byte) is below the GPU's ridge point.

What is counted:

* Weights: every linear layer, the norms and the LM head are read once per
  step; the embedding table only for the rows of the tokens in the step. The LM
  head is read (and run) only when the step emits at least one token. Weight-only
  INT8/INT4 keeps embeddings, norms and LM head in FP16, as ``weights.py`` does.
* FLOPs: ``2 x linear params`` per token, ``2 x vocab x hidden`` per emitted
  token (logits), and ``4 x layers x heads x head_dim`` per query-key pair
  (QK^T and softmax(QK^T)V, 2 FLOPs per multiply-add). Norms, softmax and
  activations are ignored (<1% of FLOPs).
* Activations are assumed to stay on chip (fused kernels, flash attention), so
  their memory traffic is not counted.
* KV swap traffic goes over PCIe and is added after the GPU work (not overlapped).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import ConfigError, GPUConfig, ModelConfig
from .footprint import kv_bytes_per_token
from .precision import FP16, Precision
from .weights import attention_params_per_layer, mlp_params_per_layer, param_breakdown

PHASES = ("prefill", "decode")


def _check_fraction(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise ConfigError(f"{name} must be in (0, 1], got {value!r}")


@dataclass(frozen=True)
class Hardware:
    """A GPU's limits as the cost model uses them.

    The efficiencies scale the datasheet peaks down to what kernels reach in
    practice. Both default to 1.0, which gives the ideal roofline: the fastest
    the GPU could possibly run the step.
    """

    gpu: GPUConfig
    bw_efficiency: float = 1.0
    compute_efficiency: float = 1.0
    step_overhead_s: float = 0.0  # kernel launches, scheduling, sampling

    def __post_init__(self) -> None:
        _check_fraction("bw_efficiency", self.bw_efficiency)
        _check_fraction("compute_efficiency", self.compute_efficiency)
        if not isinstance(self.step_overhead_s, (int, float)) or self.step_overhead_s < 0:
            raise ConfigError(f"step_overhead_s must be >= 0, got {self.step_overhead_s!r}")

    @property
    def bandwidth(self) -> float:
        """Effective HBM bytes per second."""
        return self.gpu.bandwidth_bytes_per_s * self.bw_efficiency

    @property
    def flops(self) -> float:
        """Effective FP16 FLOPs per second."""
        return self.gpu.peak_fp16_tflops * 1e12 * self.compute_efficiency

    @property
    def ridge_point(self) -> float:
        """FLOPs per byte at which compute time equals memory time."""
        return self.flops / self.bandwidth


@dataclass(frozen=True)
class StepCost:
    """Traffic, work and time of one forward pass."""

    phase: str
    tokens: int  # tokens processed (prompt tokens, or one per sequence in decode)
    logits: int  # tokens emitted (LM head runs once per emitted token)
    weight_bytes: int
    kv_read_bytes: int
    kv_write_bytes: int
    swap_bytes: int
    linear_flops: int
    attention_flops: int
    lm_head_flops: int
    t_memory_s: float
    t_compute_s: float
    t_swap_s: float
    latency_s: float

    @property
    def hbm_bytes(self) -> int:
        """Bytes moved between GPU memory and the chip."""
        return self.weight_bytes + self.kv_read_bytes + self.kv_write_bytes

    @property
    def flops(self) -> int:
        return self.linear_flops + self.attention_flops + self.lm_head_flops

    @property
    def intensity(self) -> float:
        """Arithmetic intensity in FLOPs per HBM byte."""
        return self.flops / self.hbm_bytes if self.hbm_bytes else 0.0

    @property
    def bound(self) -> str:
        return "memory" if self.t_memory_s >= self.t_compute_s else "compute"

    @property
    def tokens_per_s(self) -> float:
        """Emitted tokens per second during this step."""
        return self.logits / self.latency_s if self.latency_s else 0.0

    def row(self) -> dict:
        data = asdict(self)
        data.update(hbm_bytes=self.hbm_bytes, flops=self.flops, intensity=self.intensity,
                    bound=self.bound, tokens_per_s=self.tokens_per_s)
        return data


class PerfModel:
    """Cost model of one model with given precisions on one GPU."""

    def __init__(self, model: ModelConfig, hardware: Hardware | GPUConfig,
                 weight_precision: Precision = FP16, kv_precision: Precision = FP16) -> None:
        self.model = model
        self.hw = hardware if isinstance(hardware, Hardware) else Hardware(hardware)
        self.weight_precision = weight_precision
        self.kv_precision = kv_precision
        self.kv_bytes_per_token = kv_bytes_per_token(model, kv_precision)

        params = param_breakdown(model)
        self.linear_params = params.linear
        wp = weight_precision
        other = FP16 if wp.bits < 16 else wp  # embeddings, norms, LM head stay FP16 when quantized
        per_layer = attention_params_per_layer(model) + mlp_params_per_layer(model)
        self._linear_bytes = model.num_layers * wp.bytes_for(per_layer)
        self._norm_bytes = other.bytes_for(params.norms)
        head_params = params.lm_head or params.embedding  # tied: the embedding is the head
        self._head_bytes = other.bytes_for(head_params)
        self._row_bytes = other.bytes_for(model.hidden_size)  # one embedding row
        self._tied = params.lm_head == 0
        self._head_flops_per_token = 2 * model.vocab_size * model.hidden_size
        self._attn_flops_per_pair = 4 * model.num_layers * model.num_heads * model.head_dim

    @property
    def gpu(self) -> GPUConfig:
        return self.hw.gpu

    # --- the pieces ---
    def weight_read_bytes(self, tokens: int, logits: int) -> int:
        """Weight bytes streamed from HBM by a step over ``tokens`` tokens."""
        if tokens <= 0:
            return 0
        total = self._linear_bytes + self._norm_bytes
        if logits > 0:
            total += self._head_bytes
        if not (self._tied and logits > 0):  # tied head already brought the whole table
            total += min(tokens, self.model.vocab_size) * self._row_bytes
        return total

    def linear_flops(self, tokens: int) -> int:
        return 2 * self.linear_params * tokens

    def lm_head_flops(self, logits: int) -> int:
        return self._head_flops_per_token * logits

    def attention_flops(self, pairs: int) -> int:
        return self._attn_flops_per_pair * pairs

    # --- a whole step ---
    def step(self, phase: str, tokens: int, logits: int, kv_read_bytes: int, kv_write_bytes: int,
             attention_pairs: int, swap_bytes: int = 0) -> StepCost:
        """Cost of one forward pass described by its token and KV counts."""
        if phase not in PHASES:
            raise ConfigError(f"phase must be one of {PHASES}, got {phase!r}")
        for name, value in (("tokens", tokens), ("logits", logits), ("kv_read_bytes", kv_read_bytes),
                            ("kv_write_bytes", kv_write_bytes), ("attention_pairs", attention_pairs),
                            ("swap_bytes", swap_bytes)):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        weights = self.weight_read_bytes(tokens, logits)
        linear, attn, head = (self.linear_flops(tokens), self.attention_flops(attention_pairs),
                              self.lm_head_flops(logits))
        hbm = weights + kv_read_bytes + kv_write_bytes
        t_mem = hbm / self.hw.bandwidth
        t_comp = (linear + attn + head) / self.hw.flops
        t_swap = swap_bytes / self.gpu.pcie_bytes_per_s
        busy = tokens > 0 or swap_bytes > 0
        latency = max(t_mem, t_comp) + t_swap + (self.hw.step_overhead_s if busy else 0.0)
        return StepCost(phase, tokens, logits, weights, kv_read_bytes, kv_write_bytes, swap_bytes,
                        linear, attn, head, t_mem, t_comp, t_swap, latency)

    def decode(self, batch: int, context: int) -> StepCost:
        """One decode step: ``batch`` sequences that each already hold ``context`` tokens.

        Matches the Part 2 simulator: each sequence writes one KV entry and attention
        reads its whole cache, the new entry included.
        """
        if batch < 1 or context < 0:
            raise ValueError("decode needs batch >= 1 and context >= 0")
        bpt = self.kv_bytes_per_token
        return self.step("decode", batch, batch, batch * (context + 1) * bpt, batch * bpt,
                         batch * (context + 1))

    def prefill(self, prompt: int, batch: int = 1) -> StepCost:
        """One prefill pass over ``batch`` prompts of ``prompt`` tokens each (no chunking).

        Causal attention: token i looks at tokens 1..i, so P(P+1)/2 pairs per prompt.
        """
        if prompt < 1 or batch < 1:
            raise ValueError("prefill needs prompt >= 1 and batch >= 1")
        bpt = self.kv_bytes_per_token
        return self.step("prefill", batch * prompt, batch, 0, batch * prompt * bpt,
                         batch * prompt * (prompt + 1) // 2)

    def decode_compute_bound_batch(self, context: int, limit: int = 1 << 16) -> int | None:
        """Smallest decode batch at ``context`` that is compute-bound (None up to ``limit``)."""
        if self.decode(limit, context).bound == "memory":
            return None
        lo, hi = 0, limit  # decode(lo) memory-bound (lo=0 by convention), decode(hi) compute-bound
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.decode(mid, context).bound == "compute":
                hi = mid
            else:
                lo = mid
        return hi

    def describe(self) -> str:
        m, g = self.model, self.gpu
        eff = ""
        if self.hw.bw_efficiency != 1 or self.hw.compute_efficiency != 1:
            eff = f" at {self.hw.bw_efficiency:.0%} bandwidth / {self.hw.compute_efficiency:.0%} compute"
        return (f"{m.name} ({m.attention_type}) on {g.name}{eff} | weights "
                f"{self.weight_precision.name}, KV {self.kv_precision.name} | ridge point "
                f"{self.hw.ridge_point:.0f} FLOPs/byte")
