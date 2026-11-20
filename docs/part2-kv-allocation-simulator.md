# Part 2: KV-Cache Allocation & Memory Management Simulator

Design and implementation plan for Part 2 of the LLM KV-Cache Memory Footprint and
Bandwidth Simulator. Part 1 (`kvsim/`, up to commit `8f133f8`) is finished. This
document is the spec we build Part 2 from. It is split into milestones, and each
milestone ends in a commit with all tests passing.

| | |
|---|---|
| Question Part 2 answers | How is KV-cache memory handed out over time, and how much of it is wasted? |
| Builds on | Part 1: `kv_bytes_per_token`, `Scenario`, `memory_budget` |
| Feeds | Part 3: KV-cache size, KV bytes read and written, and batch size at every step |
| Language | Python 3.10+, standard library only (same as Part 1) |
| Status | Design agreed. Implementation follows the milestones in section 17. |

## Contents

1. [What Part 2 does](#1-what-part-2-does)
2. [How it connects to Parts 1 and 3](#2-how-it-connects-to-parts-1-and-3)
3. [OS concepts it demonstrates](#3-os-concepts-it-demonstrates)
4. [Scope](#4-scope)
5. [Design decisions](#5-design-decisions)
6. [The life of one request](#6-the-life-of-one-request)
7. [Workload](#7-workload)
8. [The KV memory pool](#8-the-kv-memory-pool)
9. [Allocators](#9-allocators)
10. [Scheduler and simulation loop](#10-scheduler-and-simulation-loop)
11. [Eviction, preemption and sliding window](#11-eviction-preemption-and-sliding-window)
12. [Metrics](#12-metrics)
13. [Outputs and the Part 3 contract](#13-outputs-and-the-part-3-contract)
14. [Command-line interface](#14-command-line-interface)
15. [Experiments for the report](#15-experiments-for-the-report)
16. [Testing and validation](#16-testing-and-validation)
17. [Milestones and commit checkpoints](#17-milestones-and-commit-checkpoints)
18. [Coordination with Parts 1 and 3](#18-coordination-with-parts-1-and-3)
19. [Risks and open points](#19-risks-and-open-points)
20. [References](#20-references)

---

## 1. What Part 2 does

Part 1 answers "how much memory does the KV-cache need?" with a formula. Part 2
answers "how is that memory handed out over time, and how much of it is wasted?"

It simulates an LLM server one step at a time. Requests arrive and wait in a queue.
When there is room, a request is admitted and its prompt is prefilled. It then
generates one token per decode step until it finishes, and its memory is released.
At every step a memory allocator has to find space for the new tokens. When memory
runs out, the simulator evicts something. It records how much KV memory is used,
reserved, wasted and free at every step.

The headline result Part 2 should show: the same GPU serves far more requests at
once when the KV-cache is handed out in small fixed-size blocks (vLLM's
PagedAttention) than when each request gets one contiguous chunk. Contiguous
allocation spends most of its memory on space reserved for tokens that are never
generated. The vLLM paper measured that existing systems used only 20.4%-38.2% of
their KV-cache memory for actual token states [1, Sec. 1].

## 2. How it connects to Parts 1 and 3

```
 Part 1                                Part 2                                      Part 3
 ------                                ------                                      ------
 ModelConfig, GPUConfig, precisions --+
 kv_bytes_per_token(model, kv_dtype) -+--> KV pool (bytes, token slots, blocks)
 memory_budget(Scenario) -------------+            |
                                                   v
 workload (random or trace CSV) --> scheduler --> allocator ----------> per-step trace --> latency per step,
                                      ^          (contiguous | paged)   (Python + CSV)     tokens/s,
                                      |               |                                    memory- vs
                                      +-- eviction <--+ overflow                           compute-bound
```

**What Part 2 uses from Part 1** (Part 1's code stays as it is, apart from one line in
`cli.py` that registers Part 2's commands; see section 18):

| Part 1 function | Used for |
|---|---|
| `load_model`, `load_gpu`, `get_precision` | the same profiles and `--kv-dtype` flags as Part 1 |
| `kv_bytes_per_token(model, kv_precision)` | bytes per token slot; block size in bytes; every byte metric |
| `Scenario`, `memory_budget(scenario)` | size of the KV pool: GPU memory minus weights, activations and reserved (section 8) |
| `model.max_context` | the longest sequence a request may have, and the contiguous "max" reservation |

**What Part 2 gives Part 3:** a per-step trace. For every step it reports the phase
(prefill or decode), the batch size, the KV-cache size, KV bytes read and written,
and query-key pairs for attention FLOPs (section 13). Part 3 turns each step into
time with `max(bytes / bandwidth, FLOPs / peak)`. Part 2 never deals in seconds.

## 3. OS concepts it demonstrates

This is the part of the project that maps most directly onto CS-339 memory
management. These pairs are worth knowing for the viva:

| In the simulator | OS concept |
|---|---|
| token slot (memory for one token's K and V) | a unit of memory |
| KV block of `B` tokens | page / frame |
| block table of a request | page table of a process |
| free block list | free frame list |
| one contiguous reservation per request | contiguous allocation (one segment per process) |
| first-fit / best-fit hole search | dynamic storage allocation |
| empty slots at the end of a request's last block | internal fragmentation |
| free memory split into holes too small to use | external fragmentation |
| preempt a request and copy its blocks to CPU RAM | swapping to a backing store |
| preempt a request and rebuild its cache later | discard the pages and rebuild them on demand |
| choosing which request to evict (LIFO / FIFO / largest) | victim selection |
| sliding window that keeps the first few tokens | ring buffer with pinned pages |
| M9: cache kept between chat turns, evicted LRU | page cache with LRU replacement, hit/miss rate |

One difference from OS paging: KV "pages" are huge. One 16-token block of LLaMA-7B
in FP16 is 16 x 512 KiB = 8 MiB, where an OS page is 4 KiB. Page-table overhead is
therefore negligible, but a half-empty last block still wastes about 4 MiB.

## 4. Scope

**Core (milestones M1-M8):**

* Token-by-token simulation: prefill (whole prompt, chunked only when it is longer
  than the per-step token budget), then one decode step per output token.
* Many requests with different prompt and output lengths. They come from a
  random generator (seeded, with presets) or from a trace CSV file.
* Continuous batching (requests join and leave at every step) and static batching
  (for comparison).
* Two allocators:
  * **Contiguous:** one reservation per request in a simulated address space, with
    first-fit or best-fit placement. The reservation strategies follow the vLLM
    paper's baselines: `max`, `request`, `pow2`, `oracle`.
  * **Paged:** fixed-size blocks (default 16 tokens), a block table per request and
    a free block list.
* Fragmentation measured at every step: used, reserved-for-later, internal and
  external fragmentation, and free memory. Compared across allocators and block sizes.
* Overflow handling for the paged allocator: whole-request preemption with LIFO
  (default), FIFO or largest-first victim choice. The evicted request is either
  recomputed (default) or swapped to CPU RAM.
* Sliding-window KV with optional attention-sink tokens.
* Per-step and per-request CSV traces, a summary table, and a Python API for Part 3.

**Later milestone (M9):** multi-turn chat sessions whose cache is kept between
turns. Cached sessions are evicted LRU or FIFO, and the simulator reports hit and
miss rates. This is where LRU means something (section 11.6).

**Stretch (only if time allows):** a buddy allocator (the vLLM paper models Orca
with one [1, Sec. 6.1]), prefix sharing with copy-on-write for parallel sampling,
mixed prefill+decode steps (chunked prefill that piggybacks on decode), and Belady's
optimal policy as a bound for M9.

**Out of scope:** time in seconds (Part 3), the effect of a sliding window on model
accuracy, real GPU kernels, multi-GPU or tensor parallelism, speculative decoding.

## 5. Design decisions

| # | Decision | Why |
|---|---|---|
| D1 | Pure Python, standard library only, same style as Part 1 (frozen dataclasses, `unittest`). | One code base; anyone in the team can run it. |
| D2 | The unit of time is the **step**: one forward pass over the current batch. | Part 2 is about memory over time. Part 3 turns steps into seconds. |
| D3 | **Continuous batching** by default; `--batching static` for comparison. | Continuous (iteration-level) batching is how servers run since Orca [2]. Static batching shows why it was introduced. |
| D4 | **Prefill and decode run in separate steps.** Prefill has priority (vLLM v0 style). A prompt longer than the token budget is prefilled over several steps. | Each step is purely prefill or purely decode, so Part 3 can label each one compute-bound or memory-bound. |
| D5 | The KV pool is sized from Part 1's budget with workspace for a **full prefill chunk** (section 8). `--pool-tokens` / `--pool-gib` override it. | Consistent with Part 1. The overrides allow small pools for demos and tests. |
| D6 | The contiguous allocator models a **real address space**: holes, first-fit or best-fit, coalescing on free. | External fragmentation then comes out of the simulation; we don't assume it. |
| D7 | Paged blocks hold `B` tokens across **all layers**; there is one block table per request. Default `B = 16`. | vLLM shares one block table across layers and defaults to 16 [1, Sec. 7.2]. |
| D8 | Eviction is **all-or-nothing per request**. Default victim: the **latest-arrived** request (LIFO). Default recovery: **recompute**; `--preempt-mode swap` copies blocks to CPU RAM. | Matches vLLM [1, Sec. 4.5]. Both recovery modes were chosen for this project. |
| D9 | The sliding window is a **per-request cap** that is always on when `--window` is set. It is not a reaction to memory being full. | That is how Mistral's rolling buffer [3] and StreamingLLM [4] use it. |
| D10 | The simulator knows each request's true output length. The scheduler and allocators only see what a real server sees (prompt length, `max_new_tokens`), except the `oracle` and `pow2` baselines. | Contiguous allocators over-reserve because the true length is unknown. That is the point of the comparison. |
| D11 | Fully deterministic: seeded random numbers; ties broken by arrival step, then request id. | The same command gives the same CSV, so tests can check exact numbers. |
| D12 | Part 2's commands live in their own module (`kvsim/cli_sim.py`) and are registered from `cli.py` with one line. | Keeps merge conflicts with Parts 1 and 3 in `cli.py` to a minimum. |

## 6. The life of one request

These are the exact rules. The unit tests check them.

* A request has a prompt of `P` tokens and generates `O` output tokens.
* **Prefill** processes the `P` prompt tokens, writes `P` KV entries and emits output
  token 1.
* Each **decode** step feeds the previously emitted token back in, writes its KV
  entry, and emits the next token.
* So `O` output tokens take 1 prefill plus `O - 1` decode steps. The cache peaks at
  `P + O - 1` entries, because the KV of the last emitted token is never written.
  A request with `O = 1` finishes in its prefill step.
* A request is valid when `P >= 1`, `O >= 1` and `P + O <= max_model_len`.

Example with `P = 4`, `O = 3`, paged allocator with `B = 4`:

| step | phase | what happens | KV entries | blocks held | output tokens |
|---|---|---|---|---|---|
| 0 | prefill | writes prompt t0-t3, emits o1 | 4 | 1 | 1 |
| 1 | decode | feeds o1 (needs a slot: new block), emits o2 | 5 | 2 | 2 |
| 2 | decode | feeds o2, emits o3, finished, memory freed | 6 | 2, then 0 | 3 |

Request states:

```
            admit (memory available)          prompt done
 WAITING ----------------------------> PREFILL -----------> DECODE ---- O tokens ----> FINISHED
   ^  |                                                       |   ^
   |  | can never fit                       preempt, recompute|   | swap in
   |  v                                                       |   |
   | REJECTED                                                 |   |
   +---------------------------------------------------------+   |
                                              preempt, swap ----> SWAPPED
```

## 7. Workload

### 7.1 Requests

```python
@dataclass(frozen=True)
class Request:
    id: int
    arrival_step: int            # step at which the request reaches the server
    prompt_len: int              # P
    output_len: int              # O, the true length (known to the simulator only)
    max_new_tokens: int | None   # limit the client asked for; default max_model_len - P
```

### 7.2 Random generator

Lengths are given as a small spec string, so the same syntax works on the command
line and in code:

| Spec | Meaning |
|---|---|
| `N` or `fixed:N` | always `N` |
| `uniform:LO:HI` | integer uniform in `[LO, HI]` |
| `lognormal:MEAN:SIGMA[:LO:HI]` | log-normal with the given mean, clipped to `[LO, HI]`; uses `random.lognormvariate(mu, SIGMA)` with `mu = ln(MEAN) - SIGMA^2 / 2` |

Presets for typical traffic. These numbers are our own choice. `short` and `chat`
are set so that `chat` has about 8.4x longer prompts and 5.8x longer outputs than
`short`, which is the ratio the vLLM paper reports between ShareGPT and Alpaca [1].

| Preset | Models | Prompt | Output |
|---|---|---|---|
| `short` | instruction following (Alpaca-like) | `lognormal:20:0.8:4:256` | `lognormal:60:0.8:4:512` |
| `chat` (default) | chatbot conversations (ShareGPT-like) | `lognormal:168:1.0:4:1536` | `lognormal:348:1.0:4:1536` |
| `long-prompt` | summarisation / RAG | `uniform:1024:1792` | `uniform:32:256` |
| `long-output` | code or story generation | `uniform:16:128` | `uniform:512:1536` |

* **Arrivals:** by default every request arrives at step 0 (an offline batch, which
  puts the most pressure on memory). `--arrival-rate R` uses a Poisson process with
  `R` requests per step on average (exponential gaps from `random.expovariate`,
  rounded down to a step).
* **Clipping:** generated output lengths are clipped so that `P + O <= max_model_len`
  and `O <= max_new_tokens`. The generator reports how many requests were clipped.
* **Seed:** `--seed` (default 0). The same seed always gives the same requests.
  Prompt lengths, output lengths and arrival gaps use three separate random streams,
  so changing one setting (for example the arrival rate in E4) leaves the other two
  exactly as they were.

### 7.3 Trace files

A plain CSV, so a workload can be edited in a spreadsheet and replayed exactly:

```csv
id,arrival_step,prompt_len,output_len,max_new_tokens
0,0,512,128,
1,0,40,900,1024
2,5,1800,64,
```

`max_new_tokens` may be empty. Loading a trace validates every row against the
model's `max_model_len` and raises `ConfigError` with the row id on bad input. It
never clips silently. `kvsim workload ... --csv trace.csv` writes a generated
workload in the same format.

### 7.4 API

```python
from kvsim.workload import (generate_workload, generate_requests, load_trace, save_trace,
                            validate_requests, workload_stats, parse_length_spec, PRESETS)

# generate_workload(n, *, preset=None, prompt=None, output=None, max_model_len,
#                   max_new_tokens=None, arrival_rate=None, seed=0) -> GeneratedWorkload
w = generate_workload(500, preset="chat", max_model_len=2048, seed=1)
w.requests, w.clipped          # list[Request], and how many were clipped to fit
reqs = generate_requests(500, prompt="uniform:16:512", output="fixed:256",   # just the list
                         max_model_len=2048, arrival_rate=0.5, seed=1)

save_trace(reqs, "trace.csv")
reqs = load_trace("trace.csv", max_model_len=2048)   # validated, sorted by (arrival_step, id)
workload_stats(reqs)           # rows of min / mean / p50 / p95 / max per length
```

`prompt` or `output` replace the preset's distribution for that one length.
`Request` also has `total_len` (`P + O`), `peak_kv_tokens` (`P + O - 1`) and
`new_token_limit(max_model_len)`. `percentile(values, p)` uses the nearest-rank
method, and the simulator's p50/p95 latencies will use the same function.

## 8. The KV memory pool

The pool is the memory left for KV-cache after everything else:

```
pool_bytes  = GPU capacity - weights - activations(one full prefill chunk) - reserved
pool_tokens = pool_bytes // kv_bytes_per_token          (token slots)
num_blocks  = pool_tokens // block_size                 (paged allocator)
block_bytes = block_size * kv_bytes_per_token
```

```python
def kv_pool_bytes(s: Scenario, max_num_seqs: int = 256) -> int:
    b = memory_budget(s.with_(context_len=s.prefill_chunk, batch=max_num_seqs))
    return b.capacity - b.weights - b.activations - b.reserved    # b.kv_cache is ignored

@dataclass(frozen=True)
class KVPool:
    bytes_per_token: int
    total_tokens: int
```

Evaluating the budget with `context_len = prefill_chunk` and `batch = max_num_seqs`
reserves activations for a full prefill chunk plus the logits of every running
sequence. That is the worst single forward pass the scheduler can issue.

**Note on Part 1's `kv_token_capacity`:** it evaluates the budget at `context_len = 0`,
`batch = 1`, which reserves activations for only one token. Our helper reserves
about 229 MiB more, which is 458 fewer token slots for LLaMA-7B in FP16. This is not
a bug in Part 1 (that function answers a different question), but the Part 1 owner
should know why the two numbers differ (section 18).

Worked numbers (FP16 KV, default 1 GiB reserved, `prefill_chunk` 2048, `max_num_seqs` 256):

| Model on GPU | KV per token | Pool | Token slots | 16-token blocks | Requests at a 2,048-token reservation |
|---|---|---|---|---|---|
| LLaMA-7B on RTX 4090 | 512 KiB | 10.23 GiB | 20,940 | 1,308 | 10 |
| LLaMA-7B on A100 80GB | 512 KiB | 66.23 GiB | 135,628 | 8,476 | 66 |
| Mistral-7B (GQA-8) on RTX 4090 | 128 KiB | 9.29 GiB | 76,065 | 4,754 | 37 |
| LLaMA-2-70B, INT4 weights, on H100 | 320 KiB | 41.75 GiB | 136,800 | 8,550 | 66 |

Quick sanity check of the headline claim: a `chat` request with a 168-token prompt
and 348 output tokens never holds more than 515 of its 2,048 reserved slots (25%),
and about 17% on average over its life. With 16-token pages it wastes at most 15
slots at any time.

If the weights alone do not fit (`pool_bytes <= 0`), the simulator stops with the
same hint Part 1 gives: try `--weight-dtype int8/int4`.

## 9. Allocators

### 9.1 Common interface

Both allocators manage the same pool and expose the same operations, so the
scheduler does not know which one it is using.

```python
class KVAllocator(ABC):
    name: str                                     # e.g. "paged-16", "contig-max-first"
    pool: KVPool
    capacity_tokens: int                          # slots it can hand out (allocated + free)

    # admission and growth; requests are identified by an integer seq_id
    def can_admit(self, seq_id, n_tokens, request=None) -> bool: ...
    def admit(self, seq_id, n_tokens, request=None) -> None: ...  # claim slots for the prompt
    def can_append(self, seq_id, n=1) -> bool: ...
    def append(self, seq_id, n=1) -> None: ...                    # claim slots for new tokens
    def trim_front(self, seq_id, n) -> None: ...                  # sliding window (added in M6)
    def free(self, seq_id) -> int: ...                            # release everything; returns slots freed

    # accounting, all in token slots
    def num_tokens(self, seq_id) -> int: ...       # slots the request has claimed
    def held_tokens(self, seq_id) -> int: ...      # slots it holds (claimed + slack)
    def allocated_tokens(self) -> int: ...         # sum over all requests
    def free_tokens(self) -> int: ...              # capacity_tokens - allocated_tokens()
    def largest_free_run(self) -> int: ...         # biggest single allocation possible now
    def check_invariants(self) -> None: ...        # used by tests after every step
    def __contains__(self, seq_id) -> bool: ...    # does the request hold memory here?
```

`request` (the `Request` from section 7) is only needed by the contiguous
allocator, which uses it to size the reservation. The paged allocator ignores it.
Calling `admit` or `append` when `can_*` is false, or using an unknown `seq_id`,
raises `AllocationError`. The scheduler always checks first.

### 9.2 Paged allocator (vLLM-style)

* The pool is cut into `num_blocks = pool_tokens // B` blocks, numbered `0..num_blocks-1`.
  The fewer than `B` slots left over cannot be used, so the paged allocator's
  `capacity_tokens` is `num_blocks x B`, slightly below `pool_tokens`.
* **Free list:** a stack of block ids. A freed block is reused first (like vLLM's
  free list), and the result is deterministic.
* **Block table:** for each request, a list of block ids. Logical block `i` holds
  token positions `[i*B, (i+1)*B)`.
* **Admit `n` tokens:** needs `ceil(n / B)` blocks. Admission also keeps a
  **watermark** of free blocks (`--watermark`, default 1% of blocks, as in vLLM v0),
  so that a new request does not cause a preemption on its very next step.
  Growth (`append`) ignores the watermark, and so do swap-ins and admissions while
  nothing else is running (otherwise a request that needs almost the whole pool
  could wait forever).
* **Append 1 token:** needs a new block only when the request's last block is full.
* **Free:** pushes all of the request's blocks back on the free list.
* **External fragmentation is zero by construction:** any free block can serve any
  request, so `largest_free_run = free_blocks * B`.
* **Metadata:** each block-table entry is a 4-byte block id, so
  `block_table_bytes = 4 x blocks in use`. It is reported but not taken from the pool,
  because it is tiny next to 8 MiB blocks.
* Each block has a reference count, always 1 in the core. It is there for the stretch
  goal of prefix sharing with copy-on-write.

### 9.3 Contiguous allocator (pre-vLLM style)

* The pool is one address space of `pool_tokens` slots. Free space is a sorted list
  of holes `(start, length)`.
* **Placement:** `--fit first` takes the lowest-address hole that is big enough (the
  default). `--fit best` takes the smallest hole that is big enough.
* **Free:** returns the region and merges it with neighbouring holes (coalescing).
* **Each request gets one region at admission and never grows.** The size of the
  region depends on the reservation strategy. The strategies mirror the baselines in
  the vLLM paper [1, Sec. 6.1]:

| `--reservation` | Reserves | Knows the true output length? | Paper's baseline |
|---|---|---|---|
| `max` (default) | `max_model_len` (2,048 for LLaMA-7B) | no | Orca (Max) |
| `request` | `P + max_new_tokens` | no | what a client-limited server does |
| `pow2` | `P + next_power_of_two(O)` (true `O` = 25 reserves 32) | yes, over-reserves at most 2x | Orca (Pow2) |
| `oracle` | `P + O - 1`, exactly the peak | yes | Orca (Oracle), the best contiguous can do |

Every reservation is capped at `max_model_len`. With a sliding window it is also
capped at `sink_tokens + window` (a ring buffer, like Mistral's rolling cache [3]).

* **External fragmentation is real here.** A request can be blocked even though
  enough memory is free in total, because no single hole is big enough. Each such
  step counts as a `frag_stall`.
* **Contiguous allocators never preempt.** Space for the whole sequence is reserved
  up front, so a running request can always grow. Memory pressure shows up as
  requests waiting in the queue (a smaller batch), not as eviction.

### 9.4 How every held slot is classified

For each request that holds GPU memory, define:

* `H`: slots held (contiguous: the reservation; paged: `blocks x B`)
* `U`: slots with a token state written in them (used)
* `F`: the most slots this request will ever need in this admission:
  `min(P + O - 1, sink_tokens + window)`

Then `H` splits into three parts, matching the three kinds of waste the vLLM paper
names [1, Sec. 3.1]:

```
used              = U
reserved_future   = min(H, F) - U      held now, will be filled later
internal_frag     = H - min(H, F)      held, will never be filled
```

For the whole pool:

```
allocated  = sum of H
free       = pool - allocated
external   = free - largest_free_run   (always 0 for paged)
```

So at every step, `used + reserved_future + internal_frag + free = pool`, and
`external_frag` is the part of `free` that cannot be used for one allocation.

## 10. Scheduler and simulation loop

### 10.1 Configuration

```python
@dataclass(frozen=True)
class SimConfig:
    allocator: str = "paged"             # "paged" | "contiguous"
    block_size: int = 16                 # paged
    watermark: float = 0.01              # paged: share of blocks kept free at admission
    reservation: str = "max"             # contiguous: max | request | pow2 | oracle
    fit: str = "first"                   # contiguous: first | best
    batching: str = "continuous"         # continuous | static
    max_num_seqs: int = 256              # most requests running at once (vLLM default)
    max_batched_tokens: int = 2048       # prompt tokens per prefill step (= Part 1's prefill_chunk)
    max_model_len: int = 2048            # from the model profile
    evict_policy: str = "lifo"           # lifo | fifo | largest   (lru arrives in M9)
    preempt_mode: str = "recompute"      # recompute | swap
    swap_bytes: int = 4 * GIB            # CPU swap space (vLLM's default swap_space is 4 GiB)
    window: int | None = None            # sliding window size in tokens
    sink_tokens: int = 0                 # first tokens always kept with a window
    max_steps: int = 10_000_000          # safety stop

def simulate(requests: list[Request], pool: KVPool, config: SimConfig) -> SimResult: ...
```

### 10.2 One step

```python
def step(self):
    self.arrivals()          # requests with arrival_step <= now join `waiting`;
                             # a request that could never fit, even alone, is REJECTED
    if nothing is running, waiting or swapped:
        jump `now` to the next arrival, and do not record idle steps
    self.swap_in()           # swap mode: oldest swapped request first, while blocks are free
    if not self.swapped:     # preempted requests go before new ones
        self.admit()         # FCFS from `waiting` (see 10.3)
    if some running request still has prompt tokens to prefill:
        self.prefill_step()  # oldest first, up to max_batched_tokens tokens in total
    else:
        self.decode_step()   # one token for every running request
    self.finish()            # requests that emitted their last token free their memory
    self.record()            # one StepStats row
    self.now += 1
```

### 10.3 Admission

* Strict **FCFS**: take the head of `waiting`. If it does not fit, stop. Later
  requests do not overtake it (head-of-line blocking, as in vLLM). Count an
  `admission_stall`, and also a `frag_stall` when total free memory would have been
  enough.
* A request is admitted only if all of these hold:
  * fewer than `max_num_seqs` requests are running
  * the step's prompt-token budget is not used up
  * `allocator.can_admit(seq, n)`, where `n` is the prompt length (or, after a
    recompute, the tokens to rebuild; section 11.3)
* Each admitted request uses up `min(n, budget left)` of this step's token budget.
  Any remaining prompt tokens are prefilled in later steps (chunked prefill).
* Memory for the whole prompt is claimed at admission. Slots that are claimed but not
  yet written count as `reserved_future`.
* **Static batching:** admission happens only when nothing is running. The batch then
  runs until every request in it has finished.

### 10.4 Prefill step

```python
budget = max_batched_tokens
for seq in prefilling requests, oldest first:
    c = min(seq.prefill_remaining, budget)          # chunk size
    kv_read_tokens  += seq.kv_len                   # earlier chunks are read back
    kv_write_tokens += c
    attention_pairs += c * seq.kv_len + c * (c + 1) // 2
    seq.kv_len += c; seq.prefill_remaining -= c; budget -= c
    if seq.prefill_remaining == 0:
        seq.generated += 1                          # prefill emits a token
    if budget == 0: break
```

### 10.5 Decode step

Slots are claimed one request at a time, so a request later in the loop sees the
blocks already taken this step:

```python
decoded = []
for seq in running, oldest first:
    if seq was preempted earlier in this step: continue
    if window and seq.kv_len == sink_tokens + window:
        allocator.trim_front(seq, 1)                # drop the oldest non-sink token; may free a block
        seq.kv_len -= 1
    overflowed = False
    while not allocator.can_append(seq):            # paged only: no free block
        if not overflowed:
            overflow_events += 1; overflowed = True
        victim = choose_victim(evict_policy, running)   # may be seq itself
        preempt(victim)                             # recompute or swap (section 11)
        if victim is seq: break
    else:
        allocator.append(seq, 1)                    # slot claimed now
        decoded.append(seq)
for seq in decoded:
    if seq is still running:                        # it may have been preempted later in the loop
        seq.kv_len += 1; seq.generated += 1
        kv_write_tokens += 1
        kv_read_tokens  += seq.kv_len               # attention reads the whole cache, new entry included
        attention_pairs += seq.kv_len
```

### 10.6 Termination

* Every admission ends its prefill with an emitted token. A request preempted during
  decode has emitted at least that token, so every admission makes progress and the
  simulation always ends.
* A request that could never fit, even alone in an empty pool, is rejected when it
  arrives. Paged: its peak block count is more than `num_blocks`. Contiguous: its
  reservation is more than `pool_tokens`.
* `max_steps` is a last-resort guard. Hitting it raises an error and means there is a
  bug.

## 11. Eviction, preemption and sliding window

### 11.1 Overflow events

An **overflow event** happens when a running request needs a new block in a decode
step and the free list is empty. It is counted once per request per step, however
many victims it takes to make room. Only the paged allocator can overflow
(section 9.3).

### 11.2 Victim policies (`--evict-policy`)

Eviction is all-or-nothing: a victim gives up all of its blocks [1, Sec. 4.5].

| Policy | Evicts | Why it is interesting |
|---|---|---|
| `lifo` (default) | the latest-arrived running request | vLLM's choice. The newest request has the least work to redo, and FCFS order is preserved. |
| `fifo` | the earliest-arrived running request | The classic replacement order. It throws away the most work (expect more recompute). |
| `largest` | the request holding the most blocks | Frees the most memory per eviction, so fewer evictions. |
| `lru` (M9) | the least recently used **cached** session | Needs multi-turn sessions (section 11.6). |

Ties are broken by arrival step, then request id.

### 11.3 What happens to a victim (`--preempt-mode`)

* **`recompute` (default):** free all of the victim's blocks and put it at the
  **front** of `waiting`. When it is admitted again, it prefills everything it knew
  before: the `L` cached entries plus the token that was waiting to be fed in
  (`L + 1` tokens). That rebuilds the cache and emits its next token.
  `recompute_tokens += L`.
* **`swap`:** copy the victim's blocks to CPU swap space
  (`swap_out_bytes += blocks x block_bytes`) and move it to `swapped`. A swapped
  request is copied back before any new admissions, as soon as there are free
  blocks for its cache plus its next token, so it is not evicted again at once
  (`swap_in_bytes`). It then carries on decoding with no second prefill. The bytes are reported so that Part 3 can turn them into
  PCIe time. If swap space is full, the victim falls back to recompute, which is
  counted.
* The vLLM paper found recompute cheaper with small blocks and swap cheaper with
  large ones, with similar results for blocks of 16-64 [1, Sec. 7.3]. Our block-size
  sweep (E3) can show the same trade-off in bytes moved against tokens recomputed.

### 11.4 Sliding window (`--window W`, `--sink-tokens S`)

* Each request keeps at most the first `S` tokens (the attention sinks, which
  StreamingLLM keeps [4]) plus the latest `W` tokens. Everything in between is
  dropped. Mistral-7B v0.1 used a window of 4,096 tokens [3].
* **Paged:** a block is kept while any of its positions is a sink or inside the
  window. When the window moves past the end of a block, that block is freed. A
  request therefore holds at most `ceil(S/B) + ceil(W/B) + 1` blocks.
* **Contiguous:** the reservation is capped at `S + W` and used as a ring buffer, so
  it never grows.
* Attention reads at most `S + W` entries per request, so `kv_read_bytes` stops
  growing too.
* Simplification: after a recompute, only the kept tokens (`S + W`) are prefilled
  again. A real model's KV depends on older tokens too, through the layers.
* The accuracy cost of dropping tokens is out of scope. We only model memory.

### 11.5 Rejection

A request that cannot fit in an empty pool is `REJECTED` when it arrives (section
10.6). The run reports it and carries on; it does not deadlock.

### 11.6 M9: multi-turn sessions and LRU

In continuous batching every running request is read at every decode step, so
"least recently used" over running requests is just FIFO. LRU means something once
some caches sit **idle**:

* **Workload:** sessions with several turns. Each turn has new prompt tokens, an
  output length and a think time (idle steps before the user's next message). The
  context of turn `k` is all earlier turns plus the new prompt.
* **Kept cache:** when a turn finishes, its KV stays in memory as `CACHED` (idle).
  If the next turn arrives while the cache is still there (a **hit**), only the new
  prompt tokens are prefilled. If the cache was evicted (a **miss**), the whole
  history is prefilled again.
* **Eviction order:** under memory pressure, cached sessions are evicted first, using
  `--cache-policy lru` (least recently finished) or `fifo` (oldest session). Only
  then are running requests preempted (11.2).
* **Optional bound:** for an idle session, the next turn's arrival step
  (`finish + think time`) is already known. Belady's optimal policy ("evict the one
  needed furthest in the future") can therefore be computed exactly and shown next
  to LRU and FIFO.
* **Metrics:** hit rate, prefill tokens saved, cache evictions.
* Tests include a crafted sequence where LRU and FIFO choose different victims, as
  in the classic OS exercise.

## 12. Metrics

**Per step** (after the step's work; all bytes are slots x `bytes_per_token`):
`kv_used_bytes`, `kv_allocated_bytes`, `reserved_future_bytes`,
`internal_frag_bytes`, `external_frag_bytes`, `free_bytes`, block counts, events
(admitted, finished, preempted, overflow, stalls, swap traffic, recompute), and the
traffic Part 3 needs. The full list is in section 13.1.

**Per request:** queueing delay, time to first token and end-to-end time, all in
steps. Also preemptions, tokens recomputed, bytes swapped, and peak KV held.

**Per run (summary):**

| Metric | Definition |
|---|---|
| `share_token_states` | sum(used) / sum(used + reserved_future + internal_frag + external_frag) over steps. The share of claimed or stranded memory that holds real tokens; comparable to the paper's 20.4%-38.2% for existing systems [1]. |
| `share_used`, `share_reserved`, `share_internal_frag`, `share_external_frag` | means over steps, as a share of the pool |
| `avg_decode_batch`, `peak_running` | mean batch size over decode steps; most requests running at once |
| `tokens_per_step` | output tokens / steps (a throughput proxy until Part 3 adds time) |
| `preemptions`, `overflow_events`, `admission_stalls`, `frag_stalls` | counts |
| `recompute_tokens`, `swap_out_bytes`, `swap_in_bytes`, `swap_fallbacks` | eviction cost (`swap_fallbacks`: swaps that became recomputes because swap space was full) |
| `queue_p50/p95`, `ttft_p50/p95`, `e2e_p50/p95` | per-request latencies in steps |

Averages are taken over recorded (busy) steps with equal weight. Prefill and decode
steps take different amounts of real time, so Part 3 can weight them by time if
needed.

## 13. Outputs and the Part 3 contract

### 13.1 Per-step trace (`--steps-csv`)

One row per busy step. The column names are a contract with Part 3, and a test
pins them (section 16). Columns marked **P3** are the ones Part 3 needs.

| Column | Meaning |
|---|---|
| `step` | step number (idle gaps are skipped, so numbers can jump) |
| `phase` **P3** | `prefill` or `decode` |
| `batch_size` **P3** | requests processed in this forward pass |
| `running`, `waiting`, `swapped` | requests in each state after the step |
| `prefill_tokens` **P3** | prompt tokens processed |
| `output_tokens` **P3** | output tokens emitted |
| `kv_tokens` | token states stored after the step |
| `kv_used_bytes` **P3** | KV-cache size after the step: `kv_tokens x bytes_per_token` |
| `kv_read_bytes` **P3** | KV bytes attention reads in this step |
| `kv_write_bytes` **P3** | KV bytes written in this step |
| `attention_pairs` **P3** | query-key pairs computed (attention FLOPs are about `4 x pairs x head_dim x num_heads x layers`) |
| `kv_allocated_bytes`, `reserved_future_bytes`, `internal_frag_bytes`, `external_frag_bytes`, `free_bytes`, `pool_bytes` | the memory breakdown (9.4) |
| `blocks_used`, `blocks_free`, `block_table_bytes` | paged only (empty for contiguous) |
| `admitted`, `finished`, `preempted`, `overflow_events` | events in this step |
| `admission_stall`, `frag_stall` | 1 if the head of the queue was blocked in this step |
| `swap_out_bytes` **P3**, `swap_in_bytes` **P3**, `swap_used_bytes` | swap traffic (PCIe) and swap space in use |
| `recompute_tokens` | KV entries rebuilt because of recompute |

Weights are not included: Part 3 adds `weight_bytes(model)` once per step, shared
by the whole batch. That sharing is why bigger batches are less memory-bound.

### 13.2 Per-request trace (`--requests-csv`)

`id, arrival_step, prompt_len, output_len, status, admitted_step, first_token_step,
finish_step, queue_steps, ttft_steps, e2e_steps, preemptions, recompute_tokens,
swap_bytes, peak_kv_tokens, peak_held_tokens`

`queue_steps = admitted_step - arrival_step` (steps spent waiting). `ttft_steps` and
`e2e_steps` count the arrival step itself, so a request that arrives, prefills and
finishes in the same step has both equal to 1. The `P = 4, O = 3` request in section 6
has `ttft_steps = 1` and `e2e_steps = 3`.

### 13.3 Python API

```python
from kvsim.config import load_model, load_gpu
from kvsim.budget import Scenario
from kvsim.kvpool import pool_from_scenario
from kvsim.simulator import SimConfig, simulate
from kvsim.workload import generate_requests

model, gpu = load_model("llama-7b"), load_gpu("rtx-4090")
pool = pool_from_scenario(Scenario(model, gpu, context_len=0))
reqs = generate_requests(500, preset="chat", max_model_len=model.max_context, seed=1)

result = simulate(reqs, pool, SimConfig(allocator="paged", block_size=16))
result.summary()           # dict, one row of the summary table
result.steps               # list[StepStats], the per-step trace
result.decode_steps()      # only the decode steps: what Part 3's decode analysis needs
result.step_rows()         # list[dict] for report.print_table / write_csv
result.request_rows()
```

How Part 3 would use it, for example:

```python
for st in result.steps:
    bytes_moved = weight_bytes(model) + st.kv_read_bytes + st.kv_write_bytes
    t = max(bytes_moved / gpu.bandwidth_bytes_per_s, flops(st) / peak_flops)
```

## 14. Command-line interface

Four new subcommands, registered from `kvsim/cli_sim.py`. They reuse Part 1's
model, GPU and precision flags (`--kv-dtype`, `--weight-dtype`, `--prefill-chunk`,
`--reserved-gib`, ...) so the numbers always agree with Part 1.

| Command | What it does |
|---|---|
| `kvsim workload` | generate a workload, print its statistics, and optionally save it as a trace CSV |
| `kvsim simulate MODEL GPU` | run one configuration; print the pool, the summary and a sample of steps |
| `kvsim compare MODEL GPU` | run several allocator configurations on the same workload; one summary row each |
| `kvsim blocksweep MODEL GPU` | run the paged allocator with a range of block sizes; one row each |

Flags:

| Group | Flags (default) |
|---|---|
| workload | `--trace FILE` or `--preset {short,chat,long-prompt,long-output}` (`chat`), `--num-requests` (200), `--prompt-len SPEC`, `--output-len SPEC`, `--max-new-tokens N`, `--arrival-rate R` (all at step 0), `--seed` (0), `--max-model-len` (model's `max_context`) |
| pool | `--pool-tokens N` or `--pool-gib X` to override the size from Part 1's budget |
| allocator | `--allocator {paged,contiguous}` (`paged`), `--block-size` (16), `--watermark` (0.01), `--reservation {max,request,pow2,oracle}` (`max`), `--fit {first,best}` (`first`) |
| scheduler | `--batching {continuous,static}` (`continuous`), `--max-num-seqs` (256), `--max-batched-tokens` (= `--prefill-chunk`) |
| eviction | `--evict-policy {lifo,fifo,largest}` (`lifo`), `--preempt-mode {recompute,swap}` (`recompute`), `--swap-gib` (4) |
| window | `--window W` (off), `--sink-tokens S` (0) |
| output | `--csv FILE` (summary), `--steps-csv FILE`, `--requests-csv FILE`, `--timeline N` (print N evenly spaced steps, default 10) |
| compare / blocksweep | `--configs contig-max,contig-pow2,contig-oracle,paged-16` / `--block-sizes 1,2,4,8,16,32,64,128,256` |

Configuration labels: `paged-<B>` and `contig-<reservation>[-best]`. The same
labels appear in the `allocator` column of every summary.

Examples:

```bash
python -m kvsim workload --preset chat --num-requests 500 --seed 1 --csv chat500.csv
python -m kvsim simulate llama-7b rtx-4090 --trace chat500.csv --steps-csv steps.csv
python -m kvsim compare llama-7b rtx-4090 --trace chat500.csv --csv compare.csv
python -m kvsim blocksweep llama-7b rtx-4090 --trace chat500.csv --csv blocks.csv
python -m kvsim simulate mistral-7b rtx-4090 --preset long-output --window 4096 --sink-tokens 4
```

## 15. Experiments for the report

Each experiment is one or two commands. The "expect" column is a hypothesis to
check, not a result.

| # | Question | How | Expect |
|---|---|---|---|
| E1 | What does growth look like for one request? | `simulate llama-7b rtx-4090 --num-requests 1 --prompt-len fixed:300 --output-len fixed:200 --steps-csv ...`, once with `--allocator contiguous` and once paged; plot `kv_used_bytes` and `kv_allocated_bytes` against `step` | Contiguous: a flat 1 GiB line from the start. Paged: a staircase of 8 MiB steps just above the used line. |
| E2 | Contiguous against paged | `compare llama-7b rtx-4090 --trace chat500.csv` | Contiguous-max: about 10 requests at once, low `share_token_states` (the paper saw 20.4%-38.2%). Paged-16: share above 90%, a much larger `avg_decode_batch`, fewer steps in total. |
| E3 | Which block size? | `blocksweep ...` with `chat`, then with `short` | Waste grows with `B` (about `B/2` slots per request); tiny blocks mean many blocks and allocations. The paper found 16-128 best on ShareGPT, and that large blocks hurt short (Alpaca) requests [1, Sec. 7.2]. |
| E4 | Eviction policies under pressure | `simulate ... --pool-gib 2 --arrival-rate 0.5`, with `--evict-policy lifo/fifo/largest` x `--preempt-mode recompute/swap` | LIFO recomputes less than FIFO; largest has the fewest preemptions; swap moves bytes instead of recomputing tokens. |
| E5 | Sliding window | `mistral-7b`, `long-output`, `--window 512/1024/2048/4096`, with and without `--sink-tokens 4` | Memory per request stops growing at the window; more requests fit; `kv_read_bytes` per step levels off. |
| E6 | External fragmentation | `--allocator contiguous --reservation request --max-new-tokens 1024`, `--fit first` against `--fit best`, `uniform` lengths | `frag_stall > 0` and `external_frag_bytes > 0` for contiguous; always 0 for paged. |
| E7 | Static against continuous batching | `--batching static` against `continuous` | Static: the batch empties out as short requests finish early, so the average batch is lower and requests wait longer. |
| E8 | With Part 3: tokens per second | feed E2's step traces into Part 3 | Paged is faster because of bigger batches (the paper reports 2-4x [1]). |
| E9 | M9: LRU against FIFO (against OPT) | multi-turn workload, `--cache-policy lru/fifo` | LRU gets a better hit rate than FIFO, and OPT bounds both. |

CSV files are git-ignored (`*.csv` in `.gitignore`). Put the result tables for the
report in `docs/part2-results.md` instead of committing CSVs.

## 16. Testing and validation

Run everything with `python -m unittest discover -s tests -t .` (39 Part 1 tests
pass today). **Every milestone commit must keep the whole suite green.**

| File | Checks |
|---|---|
| `tests/test_workload.py` | spec parsing (and errors), presets, same seed gives same requests, clipping to `max_model_len`, Poisson arrivals are non-decreasing, trace CSV round trip, invalid trace rows name the row id |
| `tests/test_kvpool.py` | pool for LLaMA-7B on RTX 4090 = 20,940 slots, 1,308 blocks of 16; the 458-slot difference from `kv_token_capacity`; overrides; error when the weights do not fit |
| `tests/test_allocator.py` | Paged (`B=4`, 8 blocks): admit 5 takes 2 blocks; tokens 6-8 take no new block; token 9 takes a 3rd; free reuses blocks LIFO; watermark; invariants. Contiguous (100 slots): A=B=C=30, free B, then a request for 35 fails with 40 free (external fragmentation 10); free A, holes merge, and 35 fits at address 0; first-fit against best-fit choose different holes; the four reservation sizes (`pow2` with O=25 reserves P+32) |
| `tests/test_simulator.py` | the P=4, O=3 timeline in section 6, row by row; O=1 finishes in its prefill step; chunked prefill (P=10, budget 4 gives chunks 4, 4, 2); every request finishes or is rejected; total output tokens = sum of `O`; the pool is empty at the end; `contig-max` never runs more than `pool // max_model_len` requests; paged runs at least as many requests at once as contiguous on the same workload; static batching admits only when nothing is running; same seed gives an identical trace |
| `tests/test_eviction.py` | a tiny pool where exactly one preemption must happen: LIFO evicts the newer request, FIFO the older, largest the bigger; recompute rebuilds `L` entries and the request still emits exactly `O` tokens; swap bytes = blocks x block_bytes; swap space full falls back to recompute; a request too big to ever fit is rejected, not deadlocked |
| `tests/test_window.py` | KV held never exceeds `S + W` tokens (paged: `ceil(S/B) + ceil(W/B) + 1` blocks); sinks are kept; contiguous reservation is capped at `S + W`; reads per step level off |
| `tests/test_cli_sim.py` | every new command runs and exits with 0; bad flags exit with 2 and an error message; the step CSV header equals the agreed column list (the Part 3 contract); `compare` prints one row per configuration |
| `tests/test_sessions.py` (M9) | hit when the next turn arrives in time; miss after eviction; a crafted case where LRU and FIFO evict different sessions |

**Invariants checked after every step** in a debug mode the tests turn on:
`used + reserved_future + internal_frag + free = pool`; no block id is both free and
held, or held twice; contiguous holes are sorted, do not overlap, and are merged; every
request's `kv_len <= held_tokens`.

**Checks against published figures** (loose bounds, to be tightened once we see the
numbers):

* `chat` on LLaMA-7B: `contig-max` has `share_token_states` below 50% (the paper saw
  20.4%-38.2% for existing systems), and `paged-16` above 90% (the paper describes
  near-zero waste). First result at M4 (500 `chat` requests on an RTX 4090):
  `contig-max` 24.9%, `contig-pow2` 39.0%, `contig-oracle` 50.1%, inside or near
  the paper's range.
* First full paged result at M5, on the same workload: `paged-16` keeps token states
  in 98.6% of its memory, averages 34.2 requests per decode step against 9.8 for
  `contig-max`, and needs 4,909 steps against 16,813 (about 3.4x fewer, in line with
  the paper's 2-4x). LIFO recomputes the fewest tokens (80,200 against 127,068 for
  FIFO), and largest-first evicts least often (92 preemptions against 384 for LIFO).
* Very short requests (20-100 tokens) bring `paged-16` down to about 86%, because a
  half-empty 16-token block is a large share of each one. 4-token blocks give about
  97%. This is the paper's observation that large blocks hurt short sequences.
* Paged waste is always under one block per running request.
* Part 1's numbers still hold: 512 KiB per token for LLaMA-7B, 128 KiB for Mistral-7B.

## 17. Milestones and commit checkpoints

Each milestone is one commit (or a few small ones) on `main`, made only when the full
test suite passes. The message names what the commit adds.

| M | Commit message | Files | Done when |
|---|---|---|---|
| M0 | Add Part 2 design doc for the KV-cache allocation simulator | `docs/part2-kv-allocation-simulator.md` | this document is committed |
| M1 | Add workload model: requests, length distributions and trace files | `kvsim/workload.py`, `tests/test_workload.py` | the generator, presets, Poisson arrivals and trace load/save are tested |
| M2 | Add KV pool sizing and the paged block allocator | `kvsim/kvpool.py`, `kvsim/allocator.py`, `tests/test_kvpool.py`, `tests/test_allocator.py` | the allocator interface plus the paged allocator (free list, block tables, watermark, invariants) are tested |
| M3 | Add contiguous allocator with first/best fit and reservation strategies | `kvsim/allocator.py`, `tests/test_allocator.py` | holes, coalescing, external fragmentation and the four reservation modes are tested |
| M4 | Add step simulator: continuous batching, prefill/decode and fragmentation metrics | `kvsim/simulator.py`, `tests/test_simulator.py` | the request timeline, chunked prefill, static/continuous batching, StepStats, RequestStats and summary are tested without eviction |
| M5 | Add preemption: LIFO/FIFO/largest victims with recompute or swap | `kvsim/eviction.py`, `kvsim/simulator.py`, `tests/test_eviction.py` | overflow events, three victim policies, recompute, swap space with fallback, and rejection are tested |
| M6 | Add sliding-window KV with attention sinks | `kvsim/allocator.py`, `kvsim/simulator.py`, `tests/test_window.py` | the window cap works for both allocators and is tested |
| M7 | Add workload/simulate/compare/blocksweep commands and CSV traces | `kvsim/cli_sim.py`, one line in `kvsim/cli.py`, `tests/test_cli_sim.py` | all commands run, CSV headers match section 13 |
| M8 | Document Part 2 and the Part 3 trace interface | `README.md` (Part 2 section, status row), `docs/part2-results.md` | experiments E1-E7 are run and their tables written up; Part 3 has the API |
| M9 | Add multi-turn sessions with LRU/FIFO cache eviction | `kvsim/sessions.py`, `kvsim/workload.py`, `kvsim/simulator.py`, `tests/test_sessions.py` | hit/miss, LRU, FIFO (and OPT) are tested; E9 is run |

Suggested calendar, following the proposal's six weeks (week 1 began 12 Nov):

| Week | Dates | Milestones |
|---|---|---|
| 2 | 19-25 Nov | M0, M1, M2 |
| 3 | 26 Nov - 2 Dec | M3, M4 |
| 4 | 3-9 Dec | M5, M6 |
| 5 | 10-16 Dec | M7, M8 (Part 3 integration happens this week) |
| 6 | 17-23 Dec | M9, final experiments, report and viva prep |

## 18. Coordination with Parts 1 and 3

**With Part 1:**

* Part 2 imports Part 1 and changes none of its code, except one line in `cli.py`
  that registers Part 2's commands. Agree on that line before M7.
* Tell the Part 1 owner about the pool-size difference in section 8 (activations for
  one token against a full prefill chunk). They may want `kv_token_capacity` to use
  the same assumption, or to document the difference.
* `README.md` is saved as UTF-16, which is why `git diff` shows it as a binary file.
  Before M8 adds the Part 2 section, agree as a team whether to convert it to UTF-8
  so that changes show up in diffs.

**With Part 3:**

* Agree on the step trace columns in section 13.1 early, ideally this week, so Part 3
  can build against a small hand-made CSV before M7 lands.
* Agree on the request timeline in section 6. In particular, decode attention reads
  the whole cache including the new entry, and the cache peaks at `P + O - 1`.
* Swap traffic needs a PCIe bandwidth to turn into time. If Part 3 wants it in the
  GPU profiles, adding an optional `pcie_gbs` field to `GPUConfig` is a shared change
  to `config.py`, so agree on it together.
* Later: if Part 3 provides a step-time function, arrivals could be given in seconds
  instead of steps. This is not needed for the core.

## 19. Risks and open points

| Risk | Mitigation |
|---|---|
| Python is too slow for big workloads | Accounting is updated incrementally, so each step costs O(running requests). 500 requests take tens of thousands of steps, which is seconds. `--timeline` and the CSV traces are optional. |
| Off-by-one errors around block boundaries and the `P + O - 1` peak | The exact timeline in section 6 is a unit test, and invariants are checked after every step. |
| Recompute thrashing (a request is admitted and preempted over and over) | The watermark at admission; the run reports `preemptions` so it is visible, not hidden. |
| Part 3 needs a column we did not plan for | The column list is one tuple in `simulator.py`, and the contract test makes any change deliberate. |
| The proposal mentions Python/C++ | Everything so far is Python. If the course needs C++, the allocators are the natural part to port: small, self-contained and hot. |

## 20. References

1. W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with
   PagedAttention", SOSP 2023. arXiv:2309.06180. Used here: Sec. 1 (20.4%-38.2% of KV
   memory holds token states in existing systems), Sec. 3.1 (reserved, internal and
   external fragmentation), Sec. 4.5 (all-or-nothing eviction, latest requests
   preempted first, swap and recompute), Sec. 6.1 (Orca Oracle/Pow2/Max baselines
   with buddy allocation), Sec. 7.2 (block size 16), Sec. 7.3 (recompute against
   swap), the abstract (2-4x throughput) and the evaluation workloads (ShareGPT
   prompts 8.4x and outputs 5.8x longer than Alpaca).
2. G.-I. Yu et al., "Orca: A Distributed Serving System for Transformer-Based
   Generative Models", OSDI 2022 (iteration-level scheduling, i.e. continuous batching).
3. A. Q. Jiang et al., "Mistral 7B", 2023. arXiv:2310.06825 (sliding-window attention
   with a 4,096-token window and a rolling buffer cache).
4. G. Xiao et al., "Efficient Streaming Language Models with Attention Sinks", 2023.
   arXiv:2309.17453 (keeping the first tokens as attention sinks plus a recent window).
5. A. Silberschatz, P. B. Galvin and G. Gagne, *Operating System Concepts*: main
   memory (contiguous allocation, fragmentation, paging) and virtual memory (page
   replacement: FIFO, LRU, optimal).
