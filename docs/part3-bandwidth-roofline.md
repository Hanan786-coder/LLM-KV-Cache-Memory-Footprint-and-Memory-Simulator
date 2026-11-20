# Bandwidth, latency and roofline model

How `kvsim` turns a forward pass into bytes, FLOPs and time. The code is in
`kvsim/perf.py` (one step), `kvsim/profiler.py` (a whole simulated run),
`kvsim/scaling.py` (sweeps) and `kvsim/plot.py` (SVG charts). Results are in
[part3-results.md](part3-results.md).

## 1. One forward pass

A step processes `T` tokens (the prompt tokens of a prefill step, or one token per
sequence in a decode step) and emits `E` tokens (one per sequence that finishes its
prefill or decodes).

### Bytes moved (HBM traffic)

```
weights  = linear layers + norms                  (always, once per step)
         + LM head                                (only if E > 0)
         + T embedding rows                       (only the rows looked up)
KV read  = every cached entry attention looks at  (from the Part 2 trace)
KV write = one entry per processed token          (from the Part 2 trace)
HBM      = weights + KV read + KV write
```

The weights are read once per step whatever the batch size. That is the whole
reason batching helps: the same bytes serve more tokens. With weight-only INT8/INT4,
the linear layers are stored with their group scales and the embeddings, norms and
LM head stay FP16, exactly as in the VRAM budget.

Activations are assumed to stay on chip (fused kernels, flash attention), so their
traffic is not counted. KV swap traffic (Part 2's `--preempt-mode swap`) goes over
PCIe (`pcie_gbs` in the GPU profile) and is added after the GPU work.

### FLOPs

```
linear    = 2 x (attention + MLP parameters) x T
LM head   = 2 x vocab x hidden x E
attention = 4 x layers x heads x head_dim x query-key pairs
```

A multiply-add is 2 FLOPs. Attention does two matrix products per query-key pair
and head (QK^T and the weighted sum of V), each `head_dim` multiply-adds. Norms,
softmax and activation functions are under 1% of the total and are ignored. For a
short context this gives the usual "2 FLOPs per parameter per token".

### Time

```
t_memory  = HBM bytes / (bandwidth x bw_efficiency)
t_compute = FLOPs     / (peak FP16 FLOP/s x compute_efficiency)
latency   = max(t_memory, t_compute) + swap bytes / PCIe bandwidth + step overhead
```

This is the roofline model: a step is limited by whichever resource it uses up
first. Its **arithmetic intensity** (FLOPs / HBM byte) tells which. Below the GPU's
**ridge point** (peak FLOP/s / bandwidth: 153 on an A100, 295 on an H100) the step
is memory-bound, above it compute-bound.

The efficiencies default to 1.0 (the ideal roofline, an upper bound on speed).
Real systems reach roughly 70-90% of peak bandwidth and 50-70% of peak FLOP/s;
`--bw-efficiency`, `--compute-efficiency` and `--step-overhead-us` let you
calibrate.

## 2. Why decode is memory-bound and prefill compute-bound

For FP16 weights, a decode step of batch `B` does `2 x params x B` FLOPs over
`2 x params` bytes of weights, an intensity of about `B`. Batch 1 does about one FLOP
per byte, more than 100x below the ridge point, so the GPU spends almost all of its
time waiting for memory and tokens/s ~= bandwidth / model size.

A prefill of `P` prompt tokens does `2 x params x P` FLOPs over the same weight
bytes, an intensity of about `P`. Prompts longer than about the ridge point are
compute-bound.

The KV-cache pulls decode back down: each sequence reads its whole cache every
step, and attention does only about one FLOP per KV byte. With MHA and a context
of a thousand tokens or more, KV reads grow with the batch as fast as the FLOPs do,
so decode never reaches the ridge, however big the batch. GQA and MQA shrink the
cache, which is why they matter for long-context serving.

## 3. A simulated run in real time

Part 2's simulator counts time in steps. The profiler prices every recorded step
with the model above and lays the steps end to end:

* a request arrives at the start of the first step at or after its `arrival_step`
  (an idle server costs no time),
* **TTFT** = end of the step that emitted its first token - arrival,
* **E2E** = end of the step that emitted its last token - arrival,
* **TPOT** = (E2E - TTFT) / (output tokens - 1).

It reads the trace from a live `SimResult` or from the CSVs written by
`kvsim simulate --steps-csv/--requests-csv`, which give identical results.

## 4. Validation

`tests/test_perf.py` checks the model against known numbers:

| Check | Expected | Model |
|-------|----------|-------|
| LLaMA-7B FP16 decode, batch 1, A100 80GB | bandwidth / weight size ~= 151 tok/s | within 3% |
| FLOPs per token | ~= 2 x parameters | within 2% |
| Batch-1 FP16 decode intensity | ~= 1 FLOP/byte | 1.00 |
| Decode crossing to compute-bound, no KV | ~= ridge point (153) | within 5 |
| Prefill crossing to compute-bound | ~= ridge point | within 20 tokens |
| H100 / A100 decode speed | 3350 / 2039 GB/s | within 2% |
| INT4 weights, batch-1 decode | ~3x faster than FP16 | 2.8-3.6x |

## 5. Limits

* Ideal overlap: memory and compute are assumed to overlap fully within a step.
* No tensor or pipeline parallelism: one model on one GPU.
* Peak compute is the FP16 tensor-core peak for every precision (weight-only
  quantization dequantizes to FP16 before the matrix multiply).
* Prefill in `scaling --vary prompt` is one unchunked pass; simulated runs use the
  simulator's chunking.
