# MobileMoE-S-SFT: vLLM serving vs. mlx_lm

vllm-metal does not call the model's own attention code. It rebuilds attention from the module's parts (`q_proj`, `q_norm`, `rope`, `scale`, ...) and runs its paged Metal kernel. This check confirms that serving through vLLM gives the same tokens as plain mlx_lm.

Run on 2026-10-28 with vllm-metal's parity tool, at the settings its nightly CI uses (40 prompts, 20 greedy tokens, top-5 rule at the first divergence, request batches of 1 and 2), in bfloat16:

```bash
.venv/bin/python third_party/vllm-metal/tools/check_parity.py \
  --model <MobileMoE-S-SFT snapshot> --trust-remote-code \
  --max-tokens 20 --top-k 5 --batch-size 1 2 --gpu-memory-utilization 0.3
```

| Prompts/request | EXACT | TOP_K_MATCH | FAIL |
|---|---:|---:|---:|
| 1 | 32 | 8 | 0 |
| 2 | 35 | 5 | 0 |

All 13 divergences swap the first and second choice. At each one, mlx_lm's log-probabilities for the two tokens are 0.000–0.312 apart (bfloat16 log-probabilities come in steps of 0.0625), so these are rounding near-ties, not model errors. A missing Q/K norm or a wrong RoPE would produce wrong text, not near-ties.

Versions: vllm 0.32.0, vllm-metal 0.32.0.dev20261028121433 (wheel), mlx 0.32.4, mlx-lm fork commit `854017b` (branch `mobilemoe`; run before the commit, on the same code), model revision `4decae01abfeaf46046b8774fdc4404b3e33a3a4`.
