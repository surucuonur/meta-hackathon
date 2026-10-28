# MobileMoE-S on vLLM Metal: where does the memory bandwidth go?

An indie-track project for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026).

**The question:** when vLLM serves Meta's on-device MobileMoE-S model on an Apple M1 Pro, how much of the GPU's memory bandwidth does it actually use, where does the remaining time go, and which fixes turn unused bandwidth into more tokens per second?

Each measured decode step is plotted against two limits: peak bandwidth (~200 GB/s) and peak compute (~5.2 TFLOPS). One request at a time should be limited by bandwidth. Running more requests per step reuses the weights already read, until compute becomes the limit. In a mixture-of-experts model, each extra request may also pull in different experts, so the bytes read grow too. Terms are defined in [GLOSSARY.md](GLOSSARY.md).

**Status:** setup. The results, roofline chart and bottleneck table will go here.

## How Meta's models are used

- **MobileMoE-S** (open weights, `facebook/MobileMoE-S-SFT`, with `-QAT` int4 later) is the model under study, served by vLLM through vllm-metal.
- **Muse Spark** (Meta Model API) reads the profiler output and the relevant engine source code and explains each bottleneck. Its explanations are saved in `analysis/findings/`.

## Layout

```
GLOSSARY.md        shared vocabulary
docs/adr/          decisions and why
third_party/       pinned submodules: vllm-metal and mlx-lm (forks, where changes go), vLLM v0.32.0 (read-only)
bench/             benchmark sweep, roofline calculation, port correctness check
analysis/          map of the hot code path; Muse Spark findings
results/runs/      every benchmark run, committed
results/traces/    Metal GPU traces (large, not committed)
```

## Setup

Requirements: Apple Silicon, macOS 15 or later, full Xcode with macOS SDK 26.2 or later, and Hugging Face access to `facebook/MobileMoE-S-SFT`.

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/surucuonur/meta-hackathon.git
cd meta-hackathon/third_party/vllm-metal
./install.sh && source .venv-vllm-metal/bin/activate
uv pip install -e ".[dev]" -e ../mlx-lm
```

`.env` (not committed) holds `MODEL_API_KEY` for Muse Spark.
