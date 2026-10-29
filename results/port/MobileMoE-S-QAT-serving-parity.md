# MobileMoE-S-QAT (int4): vLLM serving vs. mlx_lm

The same check as [for the SFT model](MobileMoE-S-SFT-serving-parity.md), on the int4 model served in the baseline: the MLX copy with bfloat16 scales made by `bench/convert_qat.py`. vllm-metal rebuilds attention around the port's quantized projections, so this confirms that serving through vLLM gives the same tokens as plain mlx_lm.

Run on 2026-10-29 with vllm-metal's parity tool at its nightly CI settings (40 prompts, 20 greedy tokens, top-5 rule at the first divergence, request batches of 1 and 2):

```bash
HF_HUB_OFFLINE=1 .venv/bin/python third_party/vllm-metal/tools/check_parity.py \
  --model models/MobileMoE-S-QAT-mlx-bfloat16 --trust-remote-code \
  --max-tokens 20 --top-k 5 --batch-size 1 2 --gpu-memory-utilization 0.3
```

| Prompts/request | EXACT | TOP_K_MATCH | FAIL |
|---|---:|---:|---:|
| 1 | 27 | 13 | 0 |
| 2 | 29 | 11 | 0 |

All 24 divergences are near-ties: at each one, mlx_lm's log-probabilities for the two tokens are 0.000–0.188 apart (bfloat16 log-probabilities come in steps of 0.0625), and vLLM scores the same two tokens within the same range. 23 swap the first and second choice; in one, vLLM picks mlx_lm's third choice, 0.0625 below its first, where vLLM gives both tokens the same score. There are more near-ties than for the SFT model (13), but none is a clear disagreement.

Versions: vllm 0.32.0, vllm-metal 0.32.0.dev20261028121433 (wheel), mlx 0.32.4, mlx-lm fork commit `d228505` (branch `mobilemoe`; run before the commit, on the same code), model `facebook/MobileMoE-S-QAT` revision `afda132cad380ac47da5ef055f186884d1c12f65`.
