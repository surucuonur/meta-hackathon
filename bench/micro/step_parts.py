"""Investigation microbenchmarks: where does a MobileMoE-S decode step's GPU time go?

Plain MLX + the mlx_lm port (no vLLM). Run from the repo root with .venv/bin/python.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

MODEL = Path("models/MobileMoE-S-QAT-mlx-bfloat16")
ROWS = [1, 2, 4, 8, 16, 32]
out = {}


def med_ms(fn, reps=15, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


model, config = load_model(MODEL)
mx.eval(model.parameters())
layers = model.model.layers
norm = model.model.norm
D = config["hidden_size"]
print("loaded", flush=True)

# 1. Dispatch floor: a chain of tiny dependent elementwise ops.
x1 = mx.ones((1, 1, D), mx.bfloat16)


def chain(n):
    y = x1
    for _ in range(n):
        y = y * 1.0001
    mx.eval(y)


per_op_us = (med_ms(lambda: chain(1000)) - med_ms(lambda: chain(10))) / 990 * 1e3
out["tiny_op_us"] = per_op_us
print(f"tiny dependent op: {per_op_us:.2f} us each", flush=True)

# 2. One big int4 matrix-vector product: the output layer (61.6 MB).
emb = model.model.embed_tokens
for R in (1, 32):
    h = mx.random.normal((1, R, D)).astype(mx.bfloat16)
    mx.eval(h)
    ms = med_ms(lambda: mx.eval(emb.as_linear(h)))
    out[f"lm_head_ms_R{R}"] = ms
    print(
        f"output layer R={R}: {ms:.3f} ms = {61.56e6 / ms / 1e6:.0f} GB/s", flush=True
    )

# 3. Full decode step, plain mlx_lm cache, context 256, synchronous and pipelined.
for R in ROWS:
    cache = make_prompt_cache(model)
    prompt = mx.random.randint(0, 100000, (R, 256))
    logits = model(prompt, cache=cache)
    tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
    mx.eval(tok)
    build, sync = [], []
    for i in range(30):
        t0, c0 = time.perf_counter(), time.thread_time()
        logits = model(tok, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        t1, c1 = time.perf_counter(), time.thread_time()
        mx.eval(tok)
        t2 = time.perf_counter()
        if i >= 5:
            build.append((c1 - c0) * 1e3)
            sync.append((t2 - t0) * 1e3)
    # Pipelined like mlx_lm generate / vllm-metal: build step n+1 while n runs.
    y = tok
    steps = 40
    t0 = time.perf_counter()
    for i in range(steps):
        logits = model(y, cache=cache)
        ny = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        mx.async_eval(ny)
        y.tolist()
        y = ny
    y.tolist()
    piped = (time.perf_counter() - t0) / steps * 1e3
    out[f"step_R{R}"] = {
        "build_cpu_ms": statistics.median(build),
        "sync_ms": statistics.median(sync),
        "pipelined_ms": piped,
    }
    print(
        f"decode step R={R:2d}: graph build CPU {statistics.median(build):.2f} ms, "
        f"build+eval {statistics.median(sync):.2f} ms, pipelined {piped:.2f} ms",
        flush=True,
    )
    del cache

# 4. Graph size of one decode step (R=1): count primitives via DOT export.
cache = make_prompt_cache(model)
logits = model(mx.random.randint(0, 100000, (1, 64)), cache=cache)
mx.eval(logits)
step = mx.argmax(model(mx.array([[5]]), cache=cache)[:, -1, :], axis=-1)
dot = Path(sys.argv[1]) / "step.dot"
mx.export_to_dot(str(dot), step)
text = dot.read_text()
out["dot_lines"] = len(text.splitlines())
print("dot lines", out["dot_lines"], flush=True)
