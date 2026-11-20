"""Number formats and their true storage cost, including quantization metadata.

Integer formats are stored in groups: every ``group_size`` consecutive values
share one scale (and, for asymmetric schemes, one zero point) kept in FP16.
That metadata is why INT4 is not exactly 4x smaller than FP16:

    bytes per value = bits / 8 + scale_bytes / group_size
    INT4, group 64, FP16 scale + zero point: 0.5 + 4/64 = 0.5625 B  (3.56x smaller)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .config import ConfigError


@dataclass(frozen=True)
class Precision:
    name: str
    bits: int
    group_size: int | None = None  # values sharing one scale; None = no metadata
    scale_bytes: int = 0  # metadata bytes stored per group (scale [+ zero point])

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ConfigError(f"{self.name}: bits must be positive")
        if self.group_size is not None and self.group_size <= 0:
            raise ConfigError(f"{self.name}: group_size must be positive")
        if self.group_size is None and self.scale_bytes:
            raise ConfigError(f"{self.name}: scale_bytes needs a group_size")

    @property
    def is_quantized(self) -> bool:
        return self.group_size is not None

    def bytes_for(self, n_values: int) -> int:
        """Exact bytes to store ``n_values`` values as one contiguous tensor."""
        if n_values < 0:
            raise ValueError("n_values must be >= 0")
        data = math.ceil(n_values * self.bits / 8)
        if not self.is_quantized:
            return data
        return data + math.ceil(n_values / self.group_size) * self.scale_bytes

    @property
    def bytes_per_value(self) -> float:
        """Average bytes per value for a large tensor, metadata included."""
        meta = self.scale_bytes / self.group_size if self.is_quantized else 0.0
        return self.bits / 8 + meta

    @property
    def effective_bits(self) -> float:
        return self.bytes_per_value * 8

    def with_group_size(self, group_size: int) -> "Precision":
        if not self.is_quantized:
            raise ConfigError(f"{self.name} is not a grouped/quantized format")
        return replace(self, group_size=group_size)


FP32 = Precision("fp32", 32)
FP16 = Precision("fp16", 16)
BF16 = Precision("bf16", 16)
FP8 = Precision("fp8", 8)  # E4M3 with a per-tensor scale; metadata is negligible
INT8 = Precision("int8", 8, group_size=128, scale_bytes=2)  # symmetric: FP16 scale
INT4 = Precision("int4", 4, group_size=64, scale_bytes=4)  # asymmetric: FP16 scale + zero

PRECISIONS = {p.name: p for p in (FP32, FP16, BF16, FP8, INT8, INT4)}


def get_precision(name: str | Precision, group_size: int | None = None) -> Precision:
    if isinstance(name, Precision):
        precision = name
    else:
        try:
            precision = PRECISIONS[name.lower()]
        except KeyError:
            raise ConfigError(
                f"unknown precision {name!r}; choose from {', '.join(PRECISIONS)}"
            ) from None
    if group_size is not None and precision.is_quantized:
        precision = precision.with_group_size(group_size)
    return precision
