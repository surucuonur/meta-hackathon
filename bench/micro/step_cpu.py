"""Investigation: split a plain-MLX decode step's CPU time into graph build and encoding.

Calling the model only records MLX operations into a lazy graph (graph build);
mx.async_eval then encodes the graph's GPU work and returns without waiting for
it. Each step is waited for before the next, so nothing overlaps.
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
for R in (1, 32):
    cache = make_prompt_cache(model)
    logits = model(mx.random.randint(0, 100000, (R, 256)), cache=cache)
    tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
    mx.eval(tok)
    build, enc, wait = [], [], []
    for i in range(40):
        c0 = time.thread_time()
        logits = model(tok, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        c1 = time.thread_time()
        mx.async_eval(tok)
        c2 = time.thread_time()
        w2 = time.perf_counter()
        mx.eval(tok)
        w3 = time.perf_counter()
        if i >= 8:
            build.append((c1 - c0) * 1e3)
            enc.append((c2 - c1) * 1e3)
            wait.append((w3 - w2) * 1e3)
    print(
        f"R={R:2d}: graph build CPU {statistics.median(build):.2f} ms | "
        f"async_eval CPU {statistics.median(enc):.2f} ms | "
        f"then GPU wait {statistics.median(wait):.2f} ms"
    )
