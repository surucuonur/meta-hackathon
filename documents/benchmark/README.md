# Baseline benchmark: MobileMoE-S on vLLM Metal

This folder holds what the baseline benchmark found and everything it measured. The sweep ran on 2026-10-29: vLLM 0.32.0 with the vllm-metal plugin, serving Meta's MobileMoE-S-QAT (int4) on an Apple M1 Pro (16-core GPU, 16 GB).

**The question.** When vLLM serves MobileMoE-S on this Mac, how much of the GPU's memory bandwidth does it use, and where does the rest of the time go?

**The answer.** It uses 12–17% of the bandwidth. A decode step produces one new token for every request in flight. At the GPU's peak bandwidth (200 GB/s) or peak compute (5.2 TFLOPS), whichever is slower, each step would take 0.9–6.2 ms; in practice each takes 5.5–40 ms, 6–8 times longer, at every setting tested. In the terms of [GLOSSARY.md](../../GLOSSARY.md), the engine is overhead-bound: the time goes neither to reading weights nor to arithmetic.

- With one request, the CPU work that prepares each step and the GPU work each need about 5 ms.
- With more requests, the GPU work itself is slow. The GPU is busy almost all the time but moves only 24–34 GB/s.

Where the time goes is analysed in [documents/findings/decode-bottlenecks.md](../findings/decode-bottlenecks.md). In short, a step is hundreds of small GPU kernels, each with a fixed cost. With many requests, the expert and attention kernels also read the same data several times.

![Roofline and step-time breakdown](roofline.png)

| Prompt tokens | Requests in flight | Step ms | Decode tokens/s | Bandwidth used | Lower bound ms |
|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 5.49 | 182 | 32.6 GB/s (16%) | 0.89 |
| 128 | 4 | 9.66 | 414 | 30.0 GB/s (15%) | 1.45 |
| 128 | 32 | 30.04 | 1,065 | 24.0 GB/s (12%) | 3.60 |
| 1024 | 1 | 5.92 | 169 | 33.3 GB/s (17%) | 0.99 |
| 1024 | 32 | 39.99 | 800 | 31.2 GB/s (16%) | 6.24 |

## Files

| File | What it holds |
|---|---|
| [findings.md](findings.md) | What the measurements show and the evidence for each finding; hypotheses not yet tested; what getting the run to work taught us; the limits of the numbers |
| [measurements.md](measurements.md) | Every number: the per-setting tables, bytes and time per step, the cost of each added request, expert counts, prompt steps, repeat agreement, all 48 client runs, GPU utilization over time, machine state and the model constants |
| [method.md](method.md) | How the run was done: machine, software, model, server and client settings, what each instrument records, how bytes, FLOPs and steady steps are counted, and how to reproduce it |
| [roofline.png](roofline.png) | The chart above, a copy of the one `bench/roofline.py` writes into the run folder |
| [gpu-utilization.png](gpu-utilization.png) | GPU utilization over time in every client run, drawn by `bench/gpu_chart.py` and explained in [measurements.md](measurements.md#gpu-utilization-over-time) |

The raw data (client results, engine step logs and server logs, about 26 MB) stays in [results/runs/2026-10-29-baseline/](../../results/runs/2026-10-29-baseline/), where the tools write it. The tables and charts here are generated from those files by `bench/roofline.py`, `bench/run_tables.py` and `bench/gpu_chart.py`.
