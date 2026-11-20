"""Shared model and GPU configuration types (the contract between Parts 1, 2 and 3).

Profiles live as JSON files under ``kvsim/profiles/{models,gpus}``. A profile can
be referenced by its key (the file name without ``.json``, case-insensitive) or
by a path to any JSON file with the same fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
MODEL_DIR = PROFILE_DIR / "models"
GPU_DIR = PROFILE_DIR / "gpus"

KIB = 1024
MIB = 1024**2
GIB = 1024**3

MLP_TYPES = {
    # number of hidden x intermediate weight matrices in one MLP block
    "swiglu": 3,  # gate, up, down (LLaMA, Mistral)
    "gelu": 2,  # up, down (GPT-2/3 style)
}


class ConfigError(ValueError):
    """Raised for an invalid or unknown model/GPU profile."""


def _require_positive_int(obj: object, *names: str) -> None:
    for name in names:
        value = getattr(obj, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{type(obj).__name__}.{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of a decoder-only transformer, as far as memory is concerned."""

    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    head_dim: int | None = None  # defaults to hidden_size // num_heads
    mlp_type: str = "swiglu"
    tie_embeddings: bool = False
    max_context: int | None = None  # longest context the model was trained for

    def __post_init__(self) -> None:
        if self.head_dim is None:
            object.__setattr__(self, "head_dim", self.hidden_size // self.num_heads)
        _require_positive_int(
            self, "num_layers", "hidden_size", "num_heads", "num_kv_heads",
            "intermediate_size", "vocab_size", "head_dim",
        )
        if self.num_kv_heads > self.num_heads or self.num_heads % self.num_kv_heads:
            raise ConfigError(
                f"{self.name}: num_heads ({self.num_heads}) must be a multiple of "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.mlp_type not in MLP_TYPES:
            raise ConfigError(f"{self.name}: mlp_type must be one of {sorted(MLP_TYPES)}")
        if self.max_context is not None:
            _require_positive_int(self, "max_context")

    @property
    def attention_type(self) -> str:
        if self.num_kv_heads == self.num_heads:
            return "MHA"
        if self.num_kv_heads == 1:
            return "MQA"
        return "GQA"

    @property
    def queries_per_kv_head(self) -> int:
        """How many query heads share one key/value head (1 for MHA)."""
        return self.num_heads // self.num_kv_heads

    def with_kv_heads(self, num_kv_heads: int) -> "ModelConfig":
        """Same model with a different number of KV heads (MHA/GQA/MQA variants)."""
        return replace(self, num_kv_heads=num_kv_heads)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GPUConfig:
    """Memory capacity, memory bandwidth and peak compute of one GPU."""

    name: str
    memory_gib: float
    bandwidth_gbs: float  # decimal GB/s, as on datasheets
    peak_fp16_tflops: float  # dense tensor-core throughput
    peak_int8_tops: float | None = None
    pcie_gbs: float = 32.0  # host link per direction (PCIe 4.0 x16), used for KV swap

    def __post_init__(self) -> None:
        for name in ("memory_gib", "bandwidth_gbs", "peak_fp16_tflops", "pcie_gbs"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ConfigError(f"GPUConfig.{name} must be a positive number, got {value!r}")

    @property
    def memory_bytes(self) -> int:
        return int(self.memory_gib * GIB)

    @property
    def bandwidth_bytes_per_s(self) -> float:
        return self.bandwidth_gbs * 1e9

    @property
    def pcie_bytes_per_s(self) -> float:
        return self.pcie_gbs * 1e9

    @property
    def ridge_point(self) -> float:
        """FLOPs per byte at which FP16 compute and memory bandwidth balance (roofline)."""
        return self.peak_fp16_tflops * 1e12 / self.bandwidth_bytes_per_s

    def to_dict(self) -> dict:
        return asdict(self)


def _from_dict(cls, data: dict, source: str):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{source}: unknown field(s) {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:  # missing required fields
        raise ConfigError(f"{source}: {exc}") from None


def _resolve(ref: str, directory: Path, kind: str) -> Path:
    path = Path(ref)
    if path.suffix == ".json" and path.is_file():
        return path
    candidate = directory / f"{ref.lower()}.json"
    if candidate.is_file():
        return candidate
    available = ", ".join(_list(directory)) or "none"
    raise ConfigError(f"unknown {kind} {ref!r}; available: {available} (or pass a .json path)")


def _list(directory: Path) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.json"))


def load_model(ref: str) -> ModelConfig:
    path = _resolve(ref, MODEL_DIR, "model")
    return _from_dict(ModelConfig, json.loads(path.read_text(encoding="utf-8")), str(path))


def load_gpu(ref: str) -> GPUConfig:
    path = _resolve(ref, GPU_DIR, "GPU")
    return _from_dict(GPUConfig, json.loads(path.read_text(encoding="utf-8")), str(path))


def list_models() -> list[str]:
    return _list(MODEL_DIR)


def list_gpus() -> list[str]:
    return _list(GPU_DIR)
