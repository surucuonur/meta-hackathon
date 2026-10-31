# Where a decode step's time goes

The baseline ([documents/benchmark/findings.md](../benchmark/findings.md)) showed that vLLM serves MobileMoE-S-QAT at 12–17% of the M1 Pro's peak bandwidth: every decode step takes 6–8 times its lower bound. It did not show where the time goes. This document does.

The evidence comes from three sources, all gathered on 2026-10-30 with the same machine, software and int4 model as the baseline:

- microbenchmarks of each suspect part, using plain MLX or vllm-metal's own kernels;
- a Python profile of vLLM serving one request;
- reading the source code to see which kernel actually runs.

No GPU trace was taken, because that needs Xcode 26.2 or later. Each claim below is marked **measured** (timed here), **from the code** (read in the source) or **estimate** (reasoned from those two).

The scripts are in [bench/micro/](../../bench/micro/), and their raw output is in [results/runs/2026-10-30-micro/](../../results/runs/2026-10-30-micro/). See [How these were measured](#how-these-were-measured).

## Summary

- **vLLM's own engine is not the problem.** Without vLLM, plain MLX runs the same model only 5–10% faster per step. The time goes to the model's GPU work as MLX runs it.
- **With one request, a step is about 870 small GPU kernels run one after another.** Each kernel has a fixed cost of a few microseconds, so the step takes about 5 ms, although reading its bytes needs only 0.9 ms.
  - Compiling the mixture-of-experts block into fewer kernels cuts that block's time by 29%.
  - vllm-metal already has an option that does this, but it doesn't recognise MobileMoE-S.
- **With many requests, two kernels read the same data more than once:**
  - **Expert matrix products** read each expert's weights once for every token that uses it. MLX only picks its kernel that reads each expert once from 60 tokens per step, so that kernel never runs during decode here. On equal data it is 25–36% faster.
  - **Decode attention** reads each key/value head once for every query head that shares it, 3 times here. MobileMoE-S's head layout is missing from the list that vllm-metal's grouped-query kernel supports. With 1,024-token prompts and 32 requests, this costs up to 6.4 ms of a 40 ms step.
- **Time to first token** is mostly waiting behind other requests' prompt steps. Prompt steps already reach 52–58% of peak compute.

## How a decode step runs

```
vllm serve        picks the requests for this step                      0.09 ms per step
   │
vllm-metal        builds the step: calls the model's layers, which only record
(model runner)    MLX operations into a graph; mx.async_eval then hands the
   │              graph to MLX and returns while the GPU runs it
   │
mlx_lm port       mobilemoe.py: 20 layers of attention + mixture of experts,
   │              written as MLX operations
   │
MLX               turns each operation into a GPU kernel (int4 matrix products,
   │              expert gathers, norms, adds, ...); vllm-metal supplies its
   │              own paged-attention kernels
   │
M1 Pro GPU        runs the kernels in order; weights stay in unified memory
```

The engine builds step n+1 on the CPU while the GPU runs step n. A step therefore takes as long as the slower of the two.

**Terms**

- **Kernel:** one program the GPU runs, such as one int4 matrix product or one elementwise add. Each kernel costs a roughly fixed amount to prepare on the CPU and to start on the GPU, whatever its size.
- **Encoding:** the CPU work of turning the graph's operations into GPU commands. In vllm-metal it happens inside `mx.async_eval`.
- **(token, expert) pair:** each decode row uses 4 of the 60 experts in every layer. A step with 32 decode rows therefore has 128 pairs per layer, but needs only 38 distinct experts ([finding 4 of the baseline](../benchmark/findings.md#4-expert-weights-become-the-largest-read-as-requests-grow)).
- **Per-pair kernel and grouped kernel:** MLX's two ways of running expert matrix products (`gather_qmm`).
  - The per-pair kernel runs one small product per pair, so an expert used by three rows is read three times.
  - The grouped kernel (`gather_qmm_rhs`) takes the pairs sorted by expert and reads each expert once.
- **Grouped-query attention (GQA):** MobileMoE-S has 12 query heads that share 4 key/value heads, 3 query heads per key/value head. A GQA kernel reads each key/value head once for all the query heads that share it.
- **`mx.compile`:** MLX traces a function once and fuses runs of elementwise operations into single kernels.

## 1. vLLM's engine adds only 5–10%

**Measured.** `step_repeats.py` runs the model with plain mlx_lm and no vLLM. Like vllm-metal, it builds each step while the GPU runs the previous one. The context is about 256 tokens, comparable to the sweep's 128-token prompts plus 256 generated tokens.

The plain MLX figures are the median of 4 repeats; the vLLM figures come from the [sweep](../benchmark/measurements.md).

| Decode rows | Plain MLX step | Plain MLX repeats | vLLM step (sweep) | vLLM's extra |
|---:|---:|---|---:|---:|
| 1 | 4.98 ms | 4.78–5.20 | 5.49 ms | +10% |
| 8 | 13.62 ms | 13.57–13.68 | 14.51 ms | +6.5% |
| 32 | 28.73 ms | 28.57–30.50 | 30.04 ms | +4.6% |

The profile agrees: vLLM's scheduler takes 0.09 ms per step. Tuning vLLM itself can recover at most about a tenth of the step. The fixes have to make the model's GPU work, and the CPU work of encoding it, cheaper.

## 2. With one request, the step is about 870 kernels in a chain

**Measured: graph size.** `step_parts.py` exports one plain-MLX decode step at 1 row (`mx.export_to_dot`). The step has 1,432 operations:

| Operations per step | Per layer | What they are (from the model code) |
|---|---:|---|
| 141 QuantizedMatmul | 7 | int4 products for q, k, v and o, plus the shared expert's gate, up and down (and 1 output layer) |
| 100 Add | 5 | two residual adds, router bias, a small epsilon, shared expert plus routed experts |
| 81 RMSNorm | 4 | two layer norms, q norm and k norm (and 1 final norm) |
| 80 AsType | 4 | type conversions, for example the router runs in float32 |
| 60 GatherQMM, 60 Arange | 3 each | expert gate, up and down products, and one index range for each |
| 40 each: Sum, SliceUpdate, RoPE, Multiply, compiled SwiGLU | 2 each | expert weighting and combining, cache writes, rotary positions, activations |
| 20 each: Sigmoid, ScaledDotProductAttention, Negative, Matmul, GatherAxis, Divide, ArgPartition | 1 each | the router (float32 product, sigmoid, top-4 selection) and attention |
| 565 Reshape, Broadcast, Transpose, Slice, Squeeze, ExpandDims, Flatten, Unflatten | | these only change how an array is viewed and launch no kernel |

The 1,432 also includes 3 Gather, 1 Quantize (the embedding lookup) and 1 ArgReduce (the next-token pick).

**Estimate: kernel count.** About **867 operations launch a GPU kernel**. This is inferred from each operation's type; a GPU trace would count them exactly. vLLM's graph differs from plain MLX in its attention and cache-write operations, but has a similar count.

**Measured: fixed cost per kernel.** A chain of 1,000 tiny dependent multiplications on a 768-element vector costs **6.81 µs per operation**. That figure is end to end: building, encoding and running each one.

**Estimate: the step is the kernel count times that cost.** 867 kernels in the 4.98 ms plain-MLX step is 5.7 µs each. Reading the step's 179 MB at peak bandwidth would take only 0.89 ms. The step is limited by the number of kernels, not by bytes or arithmetic. This is hypothesis 1 of the baseline findings; a GPU trace would confirm it.

**From the code: routing alone is about 15 of the ~43 kernels per layer.** In `MoE.__call__` (`third_party/mlx-lm/mlx_lm/models/mobilemoe.py:217-232`) these are:

- a float32 cast, the router product, sigmoid and the bias add;
- negate, argpartition and gather (the top-4 selection);
- sum, epsilon add, divide and scale (normalizing the weights);
- weight, sum, cast back and add the shared expert (combining).

## 3. Compiling the mixture-of-experts block cuts it by 29% at one row

**Measured.** `moe_parts.py` times each part of a layer, chained through all 20 layers. Each layer uses its own weights, and each feeds the next, as in a real step. The times are synchronous: they include building and encoding the graph as well as running it. The parts therefore don't add up to a step.

| Decode rows | MoE block | MoE block, compiled | Routed experts only | Shared expert | Attention projections (q, k, v, o) |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.02 ms | 4.27 ms (−29%) | 3.64 ms | 1.85 ms | 1.08 ms |
| 4 | 6.95 ms | 5.61 ms (−19%) | 3.11 ms | 2.01 ms | 1.43 ms |
| 16 | 14.03 ms | 13.45 ms (−4%) | 9.75 ms | 3.76 ms | 2.78 ms |
| 32 | 21.56 ms | 20.51 ms (−5%) | 16.60 ms | 3.89 ms | 2.92 ms |

The inputs are random vectors. Their routing spreads over more experts than the sweep's: 47.8 distinct experts per layer at 32 rows, against 38.2.

**From the code: vllm-metal already has this fix, but not for MobileMoE-S.**

- `VLLM_METAL_COMPILED_MLP=1` (off by default) wraps MLP and mixture-of-experts blocks in `mx.compile`. It only applies to decode calls of up to 4 rows (`third_party/vllm-metal/vllm_metal/compiled_mlp.py:49`, `:174-187`).
- It only wraps block classes it knows: Qwen3-Next and Qwen3.5 (`compiled_mlp.py:116-131`). MobileMoE-S's `MoE` block is not among them, so with the flag on nothing is wrapped.
- The only sign of this is that the "compiled MLP dispatch wrapped N blocks" log line never appears (`compiled_mlp.py:107-111`).
- Upstream reported 3–6% end-to-end gains from it on Qwen3.6-35B ([vllm-metal PR #621](https://github.com/vllm-project/vllm-metal/pull/621)).

**Caveat.** This measurement compiled all 20 blocks as one function. vllm-metal compiles each block separately, so its gain may be smaller.

## 4. The CPU cost per step follows the kernel count

**Measured, plain MLX** (`step_cpu.py`; each step is waited for, so nothing overlaps):

| Decode rows | Graph build (CPU) | Encoding in `mx.async_eval` (CPU) | GPU still running after encoding |
|---:|---:|---:|---:|
| 1 | 0.91 ms | 2.65 ms | 3.72 ms |
| 32 | 1.18 ms | 4.27 ms | 14.84 ms |

Encoding 867 kernels in 2.65 ms is about 3 µs per kernel (estimate).

**Measured, vLLM.** The baseline found that the submit call uses 4.73 ms of CPU per step at one request. A Python profile shows where that goes (`profile_submit.py`: one 256-token request, engine in process). Profiling slows Python code, so these figures overstate the Python parts. Per step:

- **5.91 ms in the submit call** (`_start_paged_forward`). Of that:
  - **3.63 ms inside `_submit_paged_forward_outputs`** (`third_party/vllm-metal/vllm_metal/v1/model_runner.py:891-900`). Its only heavy call is `mx.async_eval`, so this is MLX encoding the step.
  - **2.18 ms building the model graph** (`_target_forward`). Of that, 1.32 ms is in vllm-metal's attention wrapper (`attention/impls/sdpa_wrapper.py:145`).
- **0.19 ms** sampling.
- **0.09 ms** vLLM's scheduler.

Without the profiler, the same process ran at 5.63 ms per token, against 5.49 ms per step for the server in the sweep.

**Why it matters.** At one request, CPU (4.7 ms) and GPU (about 5 ms) both sit near the step time. Fewer GPU kernels alone would leave the step CPU-bound at about 4.7 ms. Fewer kernels cuts both sides at once.

## 5. With many requests, experts are read once per (token, expert) pair

**From the code.** MLX picks the grouped kernel only under the condition below (MLX v0.32.4, [mlx/backend/metal/quantized.cpp:2011-2012](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L2011-L2012)):

```cpp
  // TODO: Tune 16 and 4 here a bit better.
  if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {
```

What the condition means:

- **The variables.** `B` is the number of (token, expert) pairs and `E` the number of experts in the weight tensor, here 60 (lines 2003-2004).
- **Sorting.** `right_sorted_` is true only when mlx_lm has sorted the pairs by expert. It does so from 64 pairs, which is 16 decode rows (`third_party/mlx-lm/mlx_lm/models/switch_layers.py:181`).
- **The result.** The grouped kernel needs at least 4 × 60 = 240 pairs, which is 60 decode rows. No decode step here has more than 32.
- **What runs instead.** Every decode step uses the per-pair kernel: `gather_qvm`, because Meta's [expert, in, out] layout runs with `transpose=False` (line 2077).
- **Prompt steps** have thousands of pairs, so by the same rule they get the grouped kernel (estimate).

Sorting from 16 rows therefore does not switch kernels. It only changes the order in which the per-pair kernel visits pairs.

**Measured: the cost.**

- At 32 rows, the routed experts take 16.6 ms in the chained test, more than half of a ~29 ms step.
- The expert weights the step actually needs are 422 MB ([baseline finding 4](../benchmark/findings.md#4-expert-weights-become-the-largest-read-as-requests-grow)), which take 2.1 ms at peak bandwidth.
- The per-pair kernel is asked to read 128 × 20 × 0.553 MB = 1,416 MB.

**Measured: per-pair against grouped on equal data** (`moe_parts.py`, "kernel A/B"). The setup:

- One layer's expert weights are cut to E experts.
- The routing is constructed, not real: it is balanced, at 4–5 pairs per expert, and the pairs are in expert order in both arms.
- The gate, up and down products are chained 20 times.

| Decode rows | Pairs | Experts | Per-pair kernel | Grouped kernel | Change |
|---:|---:|---:|---:|---:|---:|
| 8 | 32 | 8 | 5.08 ms | 3.82 ms | −25% |
| 16 | 64 | 16 | 8.12 ms | 5.65 ms | −30% |
| 32 | 128 | 30 | 15.02 ms | 9.63 ms | −36% |

Real routing shares less. At 32 rows it averages 3.4 pairs per expert (128 / 38.2), and 2.2 at 16 rows. Both are below MLX's rule, and the gain there is untested.

**Measured: weight layout is not the main cause** (hypothesis 3 of the baseline). In the same test, re-quantized into MLX's usual transposed layout:

| Kernel | Transposed layout at 8 / 16 / 32 rows | Compared with Meta's layout |
|---|---|---|
| Grouped | 3.98 / 5.70 / 9.74 ms | within 4% |
| Per-pair | 4.29 / 7.61 / 14.02 ms | Meta's is 7–18% slower |

In the full model chain, Meta's layout is 5–18% faster at 1–4 rows and 7–9% slower at 16–32 rows. Changing the layout would also change the int4 numbers, because the QAT scales were trained per group along the output. It is not worth that.

## 6. At long contexts, attention reads each key/value head three times

**From the code.**

- vllm-metal's GQA decode kernel supports six layouts, written as query heads / key-value heads / head size / block size (`third_party/vllm-metal/vllm_metal/metal/ops_internal.h:47-54`):
  - 32/8/128/16
  - 24/4/256/16
  - 24/4/256/32
  - 16/2/128/16
  - 16/2/256/16
  - 16/2/256/32
- MobileMoE-S is 12/4/64/16, so the check fails (`gqa_decode_geometry_supported`, `metal/paged_attention.cpp:67-76`).
- Decode then uses the per-token kernel, which starts one GPU threadgroup per (query head, token) (`paged_attention.cpp:1026-1028`).
- Each threadgroup works out its key/value head as `head_idx / num_queries_per_kv` and reads it on its own (`metal/kernels_v2/pagedattention.metal:768`, `:860`). The 3 query heads of a group each read the same keys and values.
- Nothing logs this fallback.

**Measured.** `attention_heads.py` calls vllm-metal's attention kernel directly, the same way as its own `tools/benchmark/mixed_decode_split_benchmark.py`. The setup is 32 decode rows, block size 16 and head size 64. Each cell is the time per layer, with the rate in GB/s on the distinct key/value bytes. All three layouts ran the per-token kernel (dispatch record `per_token_ps0`).

| Context per row | 12 query / 4 KV heads (MobileMoE-S) | 4 / 4 (same cache) | 12 / 12 (3× the cache) |
|---:|---:|---:|---:|
| 256 tokens | 125 µs (67 GB/s) | 55 µs (152 GB/s) | 174 µs (145 GB/s) |
| 1,152 tokens | 569 µs (66 GB/s) | 247 µs (153 GB/s) | 820 µs (138 GB/s) |

What the table shows:

- **The cost follows the query heads, not the bytes.** 12/4 takes 2.3 times as long as 4/4 for the same cache bytes, and 69% as long as 12/12, which has three times the bytes. Some repeat reads must hit the cache, or 12/4 would be slower still.
- **The sweep agrees.** Going from 128- to 1,024-token prompts at 32 requests raises the average context from about 256 to about 1,152 tokens. The kernel predicts (569 − 125) µs × 20 layers = 8.9 ms more per step. The sweep measured 9.95 ms more (30.04 → 39.99 ms).
- **With one request, the layout barely matters:** 26 against 19 µs at 256 tokens, and 108 against 99 µs at 1,152.

**Estimate: the possible saving.** A kernel that reads each key/value head once should cost about what 4/4 costs. That gives (569 − 247) µs × 20 = up to 6.4 ms less per step at 1,024-token prompts with 32 requests. This is an upper estimate, because 4/4 also does a third of the arithmetic.

**From the code.** Even with the layout added, the GQA kernel is only admitted when the batch has enough long partitions: full partitions × query heads ≥ 33 × GPU cores (`paged_attention.cpp:156`). The gain would show mainly at long contexts.

## 7. Smaller observations

- **Output layer (measured, cause unverified).** Alone, the 61.6 MB tied output layer takes 0.931 ms at 1 row (66 GB/s) and 2.644 ms at 32 rows.
  - At 1 row that would be 17% of the step.
  - It was timed alone in short bursts, which macOS may run at lower GPU clocks ([vllm-metal issue #713](https://github.com/vllm-project/vllm-metal/issues/713)). The figure may be pessimistic.
- **Time to first token.**
  - Prompt steps already run at 52–58% of peak compute ([baseline finding 5](../benchmark/findings.md#5-steps-with-many-prompt-tokens-run-several-times-closer-to-peak-compute)).
  - Under load, the time is mostly waiting: 32 prompts of 1,024 tokens fill 16 steps of 2,048 tokens at about 0.4 s each ([finding 6](../benchmark/findings.md#6-long-prompts-stall-the-requests-already-running-for-about-04-s)).
  - Shortening that is a scheduling choice (`--max-num-batched-tokens`), not a kernel fix.

## How far this reaches

**Who pays.** Every MobileMoE-S request served by vllm-metal pays these costs, and nothing reports it. The attention fallback is silent, and the compiled-MLP option only logs when it wraps something.

**It is not specific to MobileMoE-S** (from the code, not measured):

- MLX's rule applies to every mixture-of-experts model run through `gather_qmm`. With E experts, the grouped kernel needs 4E pairs.
- The GQA list also misses Meta's Llama 3.2 1B (32/8/64) and 3B (24/8/128).
- Compiled MLP covers two model families.

**Where each fix belongs:**

| Problem | Owner |
|---|---|
| Expert kernel choice | MLX |
| Attention head layouts | vllm-metal's Metal kernels |
| Mixture-of-experts block fusion | vllm-metal's compiled-MLP policy |
| Projection fusion | the port and `bench/convert_qat.py` |

**Health:**

- **Maintainability: fragile.** A model gets the fast paths only if someone registers it in three places: a C++ table, Metal kernel instantiations and a Python class list. MLX's rule is an untuned `TODO`.
- **Correctness: healthy. Speed: at risk.** The fallbacks give correct results; the serving check found no failures in 80 requests. But they cost up to 2.3 times in attention and up to 36% in expert time, with no signal.

## What to do, in priority order

The effects below are estimates from the microbenchmarks, not end-to-end measurements. Each fix must be re-swept against the baseline ([method](../benchmark/method.md)), with its outputs checked against the baseline's.

| # | Change | Where | Expected effect | Effort | Needs Xcode |
|---|---|---|---|---|---|
| 1 | Register MobileMoE-S's `MoE` and `MLP` blocks with vllm-metal's compiled-MLP option | vllm-metal fork, Python | 1-row step from 5.5 to about 4–4.5 ms | ½ day | No |
| 2 | Fewer projection kernels: fuse q, k and v into one product; keep Meta's fused expert gate+up instead of splitting it in `convert_qat.py`; fuse the shared expert's gate and up | port and converter | 80–100 fewer kernels per step: about −0.5 ms GPU and −0.25 ms CPU | 1 day | No |
| 3 | Cut vllm-metal's per-step CPU (attention wrapper, encoding) | vllm-metal fork, Python | needed once 1 and 2 make the 1-row step CPU-bound | 1–2 days | No |
| 4 | Read each expert once per step at decode sizes: change MLX's rule in a fork, or write a grouped kernel with `mx.fast.metal_kernel` | MLX | −25–36% of expert time on equal data; about −4 to −6 ms per step at 16–32 rows | 1 day plus an MLX build, or 2–4 days | Fork: yes. Custom kernel: no |
| 5 | Add 12/4/64/16 to the GQA decode kernel | vllm-metal native code | up to −6.4 ms with 1,024-token prompts and 32 requests (+19% tokens/s) | 1–2 days | Yes |
| 6 | Scheduler settings: `--max-num-seqs 64` for throughput; `--max-num-batched-tokens 512` or `1024` to trade the 0.4 s stalls against time to first token | configuration | measured directly | about 35 min per sweep | No |

**Notes on each fix:**

1. **Compiled MLP.** This is the best gain for the effort. vllm-metal ships a parity checker for it (`tools/compiled_mlp_parity.py`). Python-only vllm-metal changes can run from the fork by reusing the wheel's compiled kernels ([README](../../README.md#setup)). The option applies only up to 4 rows, which leaves out the 4–5% measured at 16–32 rows.
2. **Projection fusion.** The joins should be exact, because they fall on whole quantization groups:
   - Dense layers group along the input, so stacking their outputs never splits a group.
   - Meta's experts group along the output, and the gate/up boundary at 384 = 12 × 32 falls between groups.

   vllm-metal's attention wrapper already accepts a packed `qkv_proj` (`attention/impls/sdpa.py:74-85`), but that is untested for this model.
3. **CPU trimming.** In the profile, the attention wrapper is 1.32 of the 2.18 ms of graph building. Fewer kernels from fixes 1 and 2 also shorten encoding.
4. **Expert kernel.** Real routing has 2.2–3.4 pairs per expert, below the test's 4–5, so the real gain is likely smaller. Whether the grouped kernel still wins at about 2 pairs per expert is untested. Changing MLX's rule affects every MoE model in MLX, so retune across sizes. A custom kernel avoids rebuilding MLX.
5. **GQA layout.** Check first that the GQA kernel template supports head size 64. The admission rule above limits the gain to long contexts.
6. **Scheduler settings.** Judge them on the client-side time to first token and 99th-percentile time between tokens, as well as on steady steps.

**Combined (estimate):**

- One-request decode drops from 5.5 to about 3.5–4 ms with fixes 1–3.
- 1,024-token prompts at 32 requests drop from 40 to about 28–30 ms with fixes 4–5.
- Reaching the 0.9 ms bandwidth floor at one request would need whole layers fused into a few kernels, which is beyond this project's deadline.

## Using the Muse Spark credits

Muse Spark (the Meta Model API) is used here only to explain, never as the model under study ([GLOSSARY.md](../../GLOSSARY.md)). This analysis changes where to point it. The slow parts are not in vLLM's own code, which adds 5–10%. They are in MLX's kernel choice, vllm-metal's per-model fast paths and the model's kernel count. Asking it about vLLM's scheduler would chase the wrong layer.

**Plan for `analysis/explain.py`** (not written yet):

1. **Inputs.** Send one focused packet per bottleneck above (sections 2, 4, 5 and 6). Each packet holds the measured numbers, the exact source excerpts cited here, and one question.
2. **Settings.** Use temperature 0 and a fixed answer format: claim, evidence lines, confidence, and the experiment that would confirm it.
   - Cache each answer by a hash of its packet.
   - Log token usage per call.
   - Read the key from `MODEL_API_KEY` in `.env`.
3. **Questions:**
   - an explanation for the report;
   - a design review: how to add 12/4/64 to the GQA kernel, what MLX's rule should be for 60 experts with 4 per token, and a sketch of a grouped decode expert kernel with `mx.fast.metal_kernel`;
   - an adversarial pass: "what else explains these numbers?", for example GPU clock behaviour.
4. **Output.** Save each answer in `analysis/findings/NN-*.md`, with a "verified by" line filled in only after a measurement confirms it. Model claims about kernel details are often wrong; the measurements decide.

**Cost.** Meta's docs list a context window of about one million tokens ([Meta Model API docs](https://ai.developer.meta.com/docs/models/)). Press reports put the price at $1.25 per million input tokens and $4.25 per million output tokens ([AI Weekly](https://aiweekly.co/alerts/meta-prices-muse-spark-11-api-at-125425-per-m-tokens), [Modelcompare](https://modelcompare.dev/models/meta/muse-spark-1-1)); this was not checked on the official pricing page. At those prices, a call with 60k tokens in and 4k out costs about $0.09, so 20 calls cost under $2.

**To settle before writing `explain.py`:**

- **Model name.** The project plan names `muse-spark-1.3`, which the docs page doesn't list. It lists 1.1, 1.2 and a 1.2 "contributor" tier.
- **Contributor tier.** It is cheaper, but prompts sent to it may be used for training.

## Not verified

- **Absolute microbenchmark times.**
  - The machine was swapping heavily: 24.8 of 25.6 GB of swap in use before the runs, and 29.5 of 30.7 GB after. Other programs were open, and short bursts on macOS can run at lower GPU clocks ([machine state](../../results/runs/2026-10-30-micro/machine.txt)).
  - The comparisons within each table were run back to back on the same data, and the attention result matches the sweep.
- **The kernel count of 867** is read from operation types, not from a GPU trace.
- **The CPU breakdown** comes from an in-process profile, which slows the Python parts.
- **Not tested at all:**
  - whether the GQA kernel template supports head size 64;
  - whether vllm-metal's packed-qkv path accepts MobileMoE-S;
  - whether the grouped expert kernel wins at real routing's 2–3.4 pairs per expert;
  - how the fixes combine.
- **Muse Spark** pricing and model name (above).

## How these were measured

The same machine and software as the baseline ([method](../benchmark/method.md)), on 2026-10-30, with no vLLM server running. All scripts load `models/MobileMoE-S-QAT-mlx-bfloat16`. "Chained over 20 layers" means each layer's output feeds the next, using each layer's own weights. Every kernel therefore waits for the previous one, as in a real step. Timings are medians of 15–21 runs after 3–4 warm-up runs.

| Script in [bench/micro/](../../bench/micro/) | What it measures | Output in [results/runs/2026-10-30-micro/](../../results/runs/2026-10-30-micro/) |
|---|---|---|
| `step_parts.py` | tiny-operation cost, the output layer, the plain-MLX decode step, the step's graph | `step_parts.txt`, `step.dot`, `step-primitives.txt` |
| `step_repeats.py` | plain-MLX pipelined step, 4 repeats | `step_repeats.txt` |
| `step_cpu.py` | graph build and encoding CPU time | `step_cpu.txt` |
| `moe_parts.py` | MoE parts chained over 20 layers; per-pair against grouped expert kernel; both weight layouts | `moe_parts.txt` |
| `attention_heads.py` | vllm-metal's decode attention for three head layouts | `attention_heads.txt` |
| `profile_submit.py` | Python profile of vLLM serving one request in process | `profile_submit.log` |

To reproduce, run from the repo root:

```bash
R=results/runs/<date>-micro && mkdir -p $R
.venv/bin/python bench/micro/step_parts.py $R
.venv/bin/python bench/micro/step_repeats.py
.venv/bin/python bench/micro/step_cpu.py
.venv/bin/python bench/micro/moe_parts.py
.venv/bin/python bench/micro/attention_heads.py
.venv/bin/python bench/micro/profile_submit.py $R
grep -v -- "->" $R/step.dot | grep -o 'label ="[A-Za-z]*' | sed 's/label ="//' | sort | uniq -c | sort -rn
```
