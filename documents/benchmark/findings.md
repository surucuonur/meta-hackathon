# Findings from the baseline benchmark

What the 2026-10-29 sweep shows about vLLM (with vllm-metal) serving MobileMoE-S-QAT on the M1 Pro. Every number below is in [measurements.md](measurements.md); how it was measured is in [method.md](method.md).

A few terms used throughout (full definitions in [GLOSSARY.md](../../GLOSSARY.md)):

- **Decode step:** one engine iteration that produces one new token for each request it includes. **Decode rows** are the requests it includes. In the steady steps studied here, that equals the number of requests the client keeps in flight.
- **Bytes a step must read:** the weights of the experts its rows use, the rest of the weights, and the rows' attention cache. **Achieved bandwidth** is those bytes divided by the step time.
- **Lower bound:** the time the step would take at peak bandwidth (200 GB/s) or peak compute (5.2 TFLOPS), whichever is slower.
- **Submit call:** vllm-metal's `MetalModelRunner._start_paged_forward`, the Python function that prepares a step's inputs and queues its GPU work. The engine builds the next step while the GPU runs the current one.

## 1. No decode step is limited by memory bandwidth or compute

At all 12 settings (128- or 1,024-token prompts, 1 to 32 requests):

- achieved bandwidth is 24.0–34.0 GB/s, 12–17% of peak;
- achieved compute is 0.10–0.60 TFLOPS, 2–12% of peak;
- each step takes 5.9–8.3 times its lower bound.

Arithmetic intensity stays at 3.1–24.9 operations per byte, below the ridge point of about 26. So by the roofline, every step should be limited by bandwidth, and none comes close.

The baseline is therefore overhead-bound everywhere: most of each step goes to something other than reading weights or doing arithmetic. Batching more requests does not change that ratio much; it stays at 6–8 times the lower bound from 1 request to 32.

## 2. With one request, the CPU and the GPU each need about 5 ms per step

With 128-token prompts, a step takes 5.49 ms. The lower bound is 0.89 ms: 179 MB at 200 GB/s.

- **CPU:** the submit call uses 4.73 ms of CPU time per step (86% of the step). Only 0.43 ms of the call is spent waiting.
- **GPU:** macOS reports the GPU busy for 96–98% of each of the three runs. That suggests about 5.3 ms of GPU work per step. This is an estimate: the counter is device-wide and only says the GPU had work.

Because the two overlap, the step takes as long as the slower one, and both are near 5 ms. Making only one side faster would gain little: the step can't drop below the 4.7 ms of CPU work unless that shrinks, nor below the GPU's 5 ms unless that shrinks. 1,024-token prompts look the same: 5.92 ms per step, 4.96 ms of CPU time, GPU busy 90–99%.

A later Python profile shows where the 4.7 ms of CPU time goes. Most of it is MLX encoding the step's GPU work, and the rest is building the model graph ([decode-bottlenecks.md, section 4](../findings/decode-bottlenecks.md#4-the-cpu-cost-per-step-follows-the-kernel-count)).

## 3. With more requests, the GPU work grows far faster than the bytes it reads

From 1 to 32 requests, the step grows from 5.49 to 30.04 ms (128-token prompts) and from 5.92 to 39.99 ms (1,024-token prompts).

- **Each added request costs 5.5–9.8 times what its bytes need.** It adds 0.55–1.54 ms to the step with 128-token prompts and 0.83–1.83 ms with 1,024-token prompts. The extra bytes it brings would take only 0.06–0.19 ms and 0.14–0.29 ms at peak bandwidth.
- **The CPU is not what grows.** The submit call's CPU time rises by only 0.03–0.26 ms per added request, reaching 6.89 ms at 32 requests (7.33 ms with long prompts). The rest of the call is waiting: 22.3 of 29.2 ms at 128-token prompts with 32 requests, and 31.8 of 39.1 ms with 1,024-token prompts.
- **The GPU is busy, but slow.** macOS reports it busy for 93–100% of every timed run with 4 or more requests, yet it moves only 24–33 GB/s.

From 4 requests up, the GPU work sets the pace, and that work runs far below what the hardware allows.

Batching still pays: 32 requests give 5.9 times the decode tokens per second of one request with 128-token prompts (1,065 vs 182), and 4.7 times with 1,024-token prompts (800 vs 169).

**The cost of each added request falls as the batch grows**, from 1.54 ms (1 to 2 requests) to 0.55 ms (16 to 32) with 128-token prompts. Two things change at about the same point, and this data can't separate them:

- each added request brings fewer new experts, because requests start to share them (finding 4);
- from 16 decode rows, mlx_lm sorts tokens by expert before the expert matrix products. Below that it does not (`do_sort = indices.size >= 64` in `mlx_lm/models/switch_layers.py`, with 4 experts per row). A later check showed that sorting does not switch MLX to its grouped expert kernel below 60 rows ([decode-bottlenecks.md, section 5](../findings/decode-bottlenecks.md#5-with-many-requests-experts-are-read-once-per-token-expert-pair)). That makes the first explanation the likelier one.

## 4. Expert weights become the largest read as requests grow

Each token uses 4 of the 60 experts in every layer. The number of distinct experts a step needs per layer grows more slowly than the number of requests, because requests share experts:

| Requests | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 128-token prompts | 4.0 | 7.0 | 12.6 | 19.6 | 29.1 | 38.2 |
| 1,024-token prompts | 4.0 | 7.1 | 12.6 | 18.6 | 25.4 | 32.9 |

With 32 requests, a step needs 38 of 60 experts per layer on average, not 128. No layer in any counted step needed more than 54.

What dominates the bytes depends on the workload:

- **128-token prompts, 32 requests:** expert weights are 422 of 720 MB (59%).
- **1,024-token prompts, 32 requests:** the attention cache is 755 of 1,248 MB (61%).
- **Weights read in every step regardless of batch** (attention, shared expert, router, norms and the output layer): 129 MB.

A fix that reduces expert reads matters most for many short requests. A fix that speeds up cache reads matters most for long contexts. The routing here comes from random-token prompts, so real text may need a different number of experts (see the limits below).

## 5. Steps with many prompt tokens run several times closer to peak compute

Steps that process 1,000–2,000 prompt tokens run at 2.7–3.0 TFLOPS, counting only the weight multiplications for the prompt tokens: 52–58% of peak compute. Decode steps reach 0.10–0.60 TFLOPS, so prompt steps run 4.5 to 30 times faster by this measure.

| Step | Prompt tokens | Step ms | TFLOPS |
|---|---:|---:|---:|
| 1,024-token prompt alone | 1,024 | 204 | 2.73 |
| about 2,000 prompt tokens with 12 decode rows | 2,023 | 397 | 2.71 |
| about 2,000 prompt tokens with 3 decode rows (128-token prompts) | 1,920 | 350 | 2.99 |

The same model, kernels and machine reach half of peak compute when a step carries enough work. This supports the overhead explanation for decode steps: whatever fixed costs a step or a GPU job has, a prompt step spreads them over a thousand or more tokens and a decode step over 1 to 32. It does not show which fixed cost it is.

Both figures are approximate. The step time includes whatever the engine overlapped with it, and the attention and decode-row work is not counted, which slightly understates the rate.

## 6. Long prompts stall the requests already running for about 0.4 s

With 1,024-token prompts and 16 or 32 requests, the 99th-percentile time between tokens is 394–413 ms, while the median is 27–41 ms.

The step log shows why. When new requests arrive, the engine fills a step with up to 2,048 prompt tokens (`--max-num-batched-tokens 2048`). Such a step takes 397 ms (median of 399 steps), and every request already generating waits for it.

The same arithmetic explains the time to first token. At 32 requests with 1,024-token prompts, the median is 1.46 s and the 99th percentile 6.5 s: 32 prompts of 1,024 tokens fill 16 steps of 2,048 tokens, at about 0.4 s each.

The client's tokens per second include these prompt steps, ramp-up and drain. That is why they fall below the steady decode rate: 501 vs 800 tokens/s at that setting.

This is the usual trade-off of splitting prompts across steps (chunked prefill), not a bandwidth problem. It matters because any fix judged by client-side numbers will mix decode speed with prompt scheduling. Fixes should be judged on the steady-step numbers first.

## 7. Repeat runs agree within 2%, except one run hit by swapping

Every setting ran three times in shuffled order. In 35 of the 36 timed runs, the run's median step time is within 2.0% of its setting's median.

The exception is the second run with 128-token prompts and 16 requests:

- **Step time:** 22.28 ms, against 20.82 and 21.00 ms for the other two runs (6–7% slower). Its 95th-percentile step was 40.9 ms, against about 21.7 ms.
- **Client view:** 536 tokens/s against 710–714, so the client numbers were hit much harder than the median step.
- **Machine:** this run saw the sweep's heaviest swapping, 7,567 swap-outs/s, and one GPU-busy sample of 30% early in its measured window.

Two runs with almost as much swapping (6,811 and 7,279/s, one request each) were not slower. Swapping is therefore the likely cause, not a proven one. The medians in the tables pool all three runs and are barely moved by it.

## 8. Hypotheses from the sweep

Three explanations fit the sweep's data. Microbenchmarks on 2026-10-30 then tested them ([decode-bottlenecks.md](../findings/decode-bottlenecks.md)):

1. **Fixed cost per GPU job: supported.** A step has about 870 kernels at roughly 6 µs each. A GPU trace would confirm it.
2. **Experts read per row: confirmed, and wider than stated.** It applies to every decode step up to 59 rows, not only below 16.
3. **Slow kernel for Meta's layout: not the main cause.** The layout changes expert time by −18% to +18%. Grouping the reads by expert saves 25–36%.

The table keeps the hypotheses as they were first stated.

| Hypothesis | Why it fits | How to test it |
|---|---|---|
| **A fixed cost per GPU job dominates.** Each step runs hundreds of small GPU operations (an estimate from the model code, not counted), and each may cost a launch and sync overhead regardless of its size. | The step is 6 times the lower bound even with one request. Prompt steps, which spread the same jobs over 1,000+ tokens, reach 52–58% of peak compute (finding 5). | A GPU trace: count the jobs per step and measure the gaps between them. If most jobs take about the same short time whatever their size, this is it. |
| **Below 16 rows, expert weights are read once per row instead of once per step.** Without sorting by expert, the expert matrix products may fetch the same expert's weights for each row that uses it. | Each added request costs 1.2–1.8 ms below 16 rows and 0.55–0.83 ms above (finding 3), and mlx_lm sorts only from 16 rows. | A GPU trace: bytes read by the expert jobs at 4 and 8 rows, compared with distinct experts times expert size. Or time the unsorted and sorted paths on the same input. |
| **The int4 expert kernel for Meta's weight layout is slow.** Meta stores experts as [expert, in, out], so the port calls MLX's `gather_qmm` with `transpose=False`, a less common path than the usual transposed one. | Expert weights are the largest read at many short requests (finding 4), and the GPU moves only 24–34 GB/s. | Needs no trace: time `gather_qmm` with `transpose=True` and `transpose=False` on the same shapes and data. |

The CPU side has a separate open question: where the 4.7 ms of the submit call goes at one request (finding 2). A Python profile of the submit call would answer it, and needs no Xcode.

## 9. What getting the run to work taught us

These are not results about speed. They shaped how the run was done.

- **The full-precision model could not be served on this machine.** vllm-metal refuses to start if macOS is swapping out memory while it reserves the attention cache. The 16 GB Mac was already swapping before the sweep: 16.9 GB of swap in use, and 322 swap-outs/s while idle. With the bfloat16 SFT model (2.5 GB), the server refused to start, first while Docker's VM held 12 GB and again after Docker quit (126 MiB still swapped out). The int4 QAT model (760 MB after conversion) started, so it became the baseline ([ADR 0002](../../docs/adr/0002-int4-qat-baseline.md)).
- **The default memory setting gave too little attention cache.** `--gpu-memory-utilization 0.3` left room for 5,504 tokens, but the largest setting needs 32 × (1,024 + 256) = 40,960. A fixed `--kv-cache-memory-bytes` of 1 GiB gives 52,416 tokens.
- **Meta's int4 checkpoint needed converting for MLX.** It stores signed 4-bit values with one float16 scale per 32 weights, and its experts are laid out [expert, in, out]. The conversion to MLX's format is exact (`bench/convert_qat.py`). The serving copy keeps its scales in bfloat16, because MLX turns bfloat16 activations times float16 scales into float32 work. That rounding is why the bfloat16 correctness gate shows an error ratio of 1.43, against a limit of 1.5.
- **Counting experts disturbs the timing.** It forces each step's GPU work to finish so the router's choices can be read. That adds 2.5–4.2 ms per step and drops GPU busy to 68–93% on average, which the [GPU chart](measurements.md#gpu-utilization-over-time) shows clearly. So expert counts come from a separate pass of the grid that is not used for timing.
- **"Did the Mac swap?" is the wrong question.** The idle machine swaps too, so a yes/no flag is useless. Each run records its own swap-out rate, compared with the idle rate.
- **The port is correct before any speed number counts.** The int4 model matches Meta's PyTorch code (float32 gate: logits within 0.0001). Serving through vLLM gives the same tokens as plain mlx_lm in all 80 requests checked, apart from near-ties ([parity report](../../results/port/MobileMoE-S-QAT-serving-parity.md)).

## 10. Limits of these numbers

- **Swapping:** 0–7,567 swap-outs/s during runs (322/s idle), with 24–40% of memory free before each run. One run was visibly affected (finding 7).
- **Bytes:** activations are not counted, and each distinct expert is counted once per step: the minimum a step must read. Real traffic is somewhat higher, so the achieved bandwidth here is a lower estimate.
- **GPU busy:** macOS's device-wide counter, sampled once per second. It says the GPU had work, not how well that work ran. The readings have no timestamps, so their times are estimated to about a second. 9 of the 816 readings in the timed windows are below 50%; [measurements.md](measurements.md#gpu-utilization-over-time) shows where they fall.
- **Prompts:** random tokens. Expert routing on real text may need more or fewer distinct experts.
- **Scope:** one machine, one software version, and greedy decoding with 256 output tokens per request.

## Next steps

Items 2 and 3 below are done; the results and a prioritized fix list are in [decode-bottlenecks.md](../findings/decode-bottlenecks.md).

1. **GPU trace** at 1 and 16 requests, to confirm the kernel count and per-kernel cost. This needs full Xcode 26.2 or later.
2. ~~Time `gather_qmm` in both weight layouts.~~ Done.
3. ~~Python profile of the submit call at one request.~~ Done.
4. **Fixes, one at a time**, each measured against this baseline with the same sweep.
