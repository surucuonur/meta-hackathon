# How the baseline benchmark was run

How the numbers in [measurements.md](measurements.md) were produced. The tools are `bench/sweep.py`, `bench/steplog/`, `bench/roofline.py` and `bench/run_tables.py`; setup is in the main [README](../../README.md#setup).

## Machine

MacBookPro18,1: Apple M1 Pro with a 16-core GPU and 16 GB of unified memory, macOS 15.7.4 (24G517), on AC power with Low Power Mode off.

The limits are the manufacturer's ratings, not measured here:

- **Peak bandwidth:** 200 GB/s.
- **Peak compute:** 5.2 TFLOPS.
- **Ridge point:** these two meet at about 26 operations per byte.

## Software

| Component | Version | Commit |
|---|---|---|
| vLLM | 0.32.0 (CPU wheel; vllm-metal supplies the Metal backend) | `5df63a7` (`third_party/vllm`, read-only) |
| vllm-metal | 0.32.0.dev20261028121433 (prebuilt wheel of the pinned commit) | `e177b67` |
| MLX | 0.32.4 | |
| mlx-lm | 0.32.0, the fork with the MobileMoE port (branch `mobilemoe`) | `d228505` |
| This repo, as the sweep ran | | `846bf63` |

## Model

`facebook/MobileMoE-S-QAT` at revision `afda132cad380ac47da5ef055f186884d1c12f65`, converted by `bench/convert_qat.py` into MLX's int4 format with bfloat16 scales (`models/MobileMoE-S-QAT-mlx-bfloat16`, 760 MB on disk).

- **Layers:** 20, all mixture-of-experts. Each has 60 routed experts, of which each token uses 4, plus one shared expert.
- **Attention:** 12 query heads and 4 key/value heads of size 64.
- **Vocabulary:** 128,256 tokens, with the output layer tied to the embedding.

[ADR 0002](../../docs/adr/0002-int4-qat-baseline.md) explains why the int4 model is the baseline. Before the sweep, the port passed its correctness gates against Meta's PyTorch code, and the vLLM serving check found no failures ([README](../../README.md#port-check)).

## Server

One `vllm serve` process per phase, with these settings:

| Setting | Value | Why |
|---|---|---|
| `--max-model-len` | 2048 | The longest request is 1,024 + 256 tokens. |
| `--max-num-seqs` | 32 | The largest concurrency, so the engine never holds back requests the client sends. |
| `--max-num-batched-tokens` | 2048 | The most prompt tokens one step may take. |
| `--kv-cache-memory-bytes` | 1 GiB | A fixed attention cache of 52,416 tokens. The grid needs at most 32 × 1,280 = 40,960. |
| `--no-enable-prefix-caching` | | Prompts share nothing, and every run should do the same work. |
| `--generation-config` | `vllm` | Ignore the model's own sampling defaults; the client asks for greedy decoding. |
| `--trust-remote-code` | | The model's config points to Meta's own code files. |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `1` | The engine runs in its own process, apart from the API server. |
| `HF_HUB_OFFLINE` | `1` | No network access during the sweep. |

## Workload

`vllm bench serve` with the random dataset, which generates prompts of exactly 128 or 1,024 random tokens.

- **Output:** 256 tokens per request, with the end-of-text token ignored so every request generates all 256.
- **Decoding:** greedy (temperature 0), with seed 1, so every run of a setting sends the same prompts.
- **Load:** requests are sent as fast as the concurrency limit allows (`--request-rate inf --max-concurrency c`), `max(8, 4 × c)` of them per run.
- **Warmup:** in the timed phase, one request per concurrency slot is sent before the measured run starts.
- **Grid:** prompts of 128 and 1,024 tokens × concurrency 1, 2, 4, 8, 16 and 32.

The sweep has two phases, each with its own server:

1. **Timed:** every setting runs three times. Each repeat uses a new shuffled order (`random.Random(1 + rep)`), so slow drift such as heat or background load does not line up with one setting.
2. **Experts:** every setting runs once more with expert counting on (below). These runs give expert counts, not timings.

The **measured window** of a client run starts when the client prints "Starting main benchmark run", after its warmup, and ends when the client exits.

## Instruments

| Instrument | What it records |
|---|---|
| Step log (`bench/steplog`, a vLLM plugin, active when `BENCH_STEP_LOG` is set) | One JSON line per call to vllm-metal's `MetalModelRunner._start_paged_forward`: the wall clock and a monotonic clock at the call's start, the call's wall time (`build_ms`) and the thread's CPU time inside it (`build_cpu_ms`), the decode rows (`rows`), the prompt tokens in the step (`prefill`) and the sum of the decode rows' context lengths (`ctx`). |
| Expert counter (same plugin, `BENCH_EXPERT_LOG=1`) | Also wraps mlx_lm's `SwitchGLU`, reads back which experts the router chose and stores the distinct experts per layer for the step's decode rows. It forces the GPU work to finish, adding 2.5–4.2 ms per step, so it runs only in the experts phase. |
| GPU utilization | macOS's device-wide "Device Utilization %" from `ioreg` (AGXAccelerator), once per second during each run, plus three idle samples before it. |
| Machine state, before and after each run | Swap in use (`sysctl vm.swapusage`), free memory (`memory_pressure`), swap-in, swap-out and page-out counters (`vm_stat`), and thermal warnings (`pmset -g therm`). A 30-second idle swap rate is measured before the sweep starts. |
| Client results | `vllm bench serve --save-result --save-detailed`: throughput, and time to first token, time per output token, time between tokens and request latency at the 50th, 95th and 99th percentiles. |

## From logs to numbers

**Steady decode steps.** A step counts if it has as many decode rows as the client's concurrency, carries no prompt tokens, and the steps before and after it do too. This leaves out ramp-up, drain and every step that mixes in prompt work. A step's time is its **period**: from its start to the start of the next step. With the engine building one step while the GPU runs the previous one, the period is the rate at which steps complete.

**Per setting.**

- **Step time:** the median period over all steady steps of the three repeats; the 95th percentile and each repeat's own median are kept too.
- **Decode tokens per second:** decode rows ÷ step time.

**Bytes a step must read:**

```
every-step weights                         67.6 MB
+ output layer                             61.6 MB
+ distinct experts per layer × 20 layers × 0.553 MB
+ (context tokens + decode rows) × 20,480 bytes of attention cache
+ decode rows × 480 bytes of embedding row
```

Distinct experts per layer come from the experts phase. Each distinct expert is counted once per step, which is the minimum the step must read. Activations are not counted.

**Operations per step:** decode rows × 544.9 million (2 × the weights one token uses) + context tokens × 61,440 (attention scores and weighted sums).

**Derived figures:**

- **Achieved bandwidth** = bytes ÷ step time.
- **Achieved compute** = operations ÷ step time.
- **Arithmetic intensity** = operations ÷ bytes.
- **Lower bound** = the larger of bytes ÷ 200 GB/s and operations ÷ 5.2 TFLOPS.

**Sanity check.** Achieved bandwidth must never exceed peak bandwidth; if it did, the byte accounting would be wrong. The highest here is 34.0 GB/s.

**GPU busy during a run.** The readings carry no timestamps, so each one's time is estimated. The sampler starts right after the "before" machine state and stops when the client exits, taking a reading every second plus the time `ioreg` needs, so the readings are spread evenly over that span: one every 1.03–1.09 s in this sweep. The estimate is good to about a second. A run's GPU busy figures use the readings that fall inside its measured window, and `bench/gpu_chart.py` draws all of them over time.

**Swap rate of a run** is the change in the swap-out counter from just before the client starts to just after it ends, divided by the elapsed time.

## Reproduce

Needs the setup in the main [README](../../README.md#setup), the converted model in `models/`, nothing else listening on port 8000, and the machine otherwise idle. The sweep took 30 minutes.

```bash
.venv/bin/python bench/sweep.py --label baseline
.venv/bin/python bench/roofline.py results/runs/<date>-baseline
.venv/bin/python bench/run_tables.py results/runs/<date>-baseline
.venv/bin/python bench/gpu_chart.py results/runs/<date>-baseline
```
