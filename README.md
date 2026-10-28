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

Requirements: Apple Silicon, macOS 15 or later, [uv](https://docs.astral.sh/uv/), and Hugging Face access to `facebook/MobileMoE-S-SFT`. Full Xcode 26.2 or later is only needed to open GPU traces and to rebuild vllm-metal's kernels.

```bash
git clone --recurse-submodules https://github.com/surucuonur/meta-hackathon.git
cd meta-hackathon
uv venv .venv --python-preference only-managed --python cpython-3.12-macos-aarch64 --seed
source .venv/bin/activate
uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcpu-cp38-abi3-macosx_11_0_arm64.whl"
uv pip install "https://github.com/vllm-project/vllm-metal/releases/download/v0.32.0.dev20261028121433/vllm_metal-0.32.0.dev20261028121433-cp312-cp312-macosx_15_0_arm64.whl"
uv pip install --no-deps -e third_party/mlx-lm
```

Python must be native arm64. An Intel Python running under Rosetta can't install these wheels, so `--python` forces uv's own arm64 build. The vllm-metal wheel is the prebuilt release of the commit pinned in `third_party/vllm-metal`. The editable `mlx-lm` install is the fork where the MobileMoE port lives.

`.env` is not committed. It holds `MODEL_API_KEY` for Muse Spark and `HUGGINGFACE_API_KEY` for downloading the gated model; the scripts pass the second one to Hugging Face tools as `HF_TOKEN`.
