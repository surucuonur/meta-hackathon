"""Investigation: plain-MLX decode step time without vLLM, repeated to show its spread.

Pipelined like mlx_lm generate and vllm-metal: step n+1 is built and submitted
while step n runs. Four repeats per decode-row count, interleaved, each with
its own random 256-token prompts.
Run from the repo root with .venv/bin/python.
"""

import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

model, _ = load_model(Path("models/MobileMoE-S-QAT-mlx-bfloat16"))
mx.eval(model.parameters())


def run(R, seed):
    mx.random.seed(seed)
    cache = make_prompt_cache(model)
    logits = model(mx.random.randint(0, 100000, (R, 256)), cache=cache)
    y = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
    mx.eval(y)
    for _ in range(5):  # warm
        logits = model(y, cache=cache)
        y = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        mx.eval(y)
    steps = 60
    t0 = time.perf_counter()
    for i in range(steps):
        logits = model(y, cache=cache)
        ny = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        mx.async_eval(ny)
        y.tolist()
        y = ny
    y.tolist()
    return (time.perf_counter() - t0) / steps * 1e3


res = {R: [] for R in (1, 8, 32)}
for rep in range(4):
    for R in (1, 8, 32):
        res[R].append(run(R, rep))
for R, v in res.items():
    print(
        f"R={R:2d} pipelined step ms per repeat: "
        + ", ".join(f"{x:.2f}" for x in v)
        + f"  median {statistics.median(v):.2f}"
    )
