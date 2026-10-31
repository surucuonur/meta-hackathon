"""Investigation microbenchmark, part 3: does decode attention re-read K/V per query head?

Calls vllm-metal's paged attention primitive directly (as its own
tools/benchmark/mixed_decode_split_benchmark.py does) on decode-only batches.
Same cache, different head layouts.
"""

import statistics
import time

import mlx.core as mx
from vllm_metal.metal import get_ops

ops = get_ops()
BLOCK, HEAD, LAYERS = 16, 64, 20


def build(R, L, qh, kvh):
    n = -(-L // BLOCK)
    tables = [list(range(1 + i * n, 1 + (i + 1) * n)) for i in range(R)]
    shape = (1 + R * n, BLOCK, kvh, HEAD)
    x = dict(
        query=mx.random.normal((R, qh, HEAD)).astype(mx.bfloat16),
        key_cache=mx.random.normal(shape).astype(mx.bfloat16),
        value_cache=mx.random.normal(shape).astype(mx.bfloat16),
        block_tables=mx.array(tables, dtype=mx.int32),
        seq_lens=mx.array([L] * R, dtype=mx.int32),
        cu_seqlens=mx.arange(R + 1, dtype=mx.int32),
    )
    mx.eval(*x.values())
    return x


def call(x, R, L, kvh):
    kwargs = dict(num_decode_requests=R, num_decode_tokens=R, max_decode_context_len=L)
    if R > 1:
        kwargs["gqa_length_plan"] = ops.gqa_decode_length_plan([L] * R)
    return ops.paged_attention_primitive(
        x["query"],
        x["key_cache"],
        x["value_cache"],
        kvh,
        HEAD**-0.5,
        0.0,
        x["block_tables"],
        x["seq_lens"],
        x["cu_seqlens"],
        BLOCK,
        L,
        -1,
        **kwargs,
    )


def per_layer_ms(x, R, L, kvh, reps=15):
    ts = []
    for i in range(reps + 3):
        outs = [call(x, R, L, kvh) for _ in range(LAYERS)]
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(outs)
        if i >= 3:
            ts.append((time.perf_counter() - t0) * 1e3 / LAYERS)
    return statistics.median(ts)


previous = ops._set_paged_dispatch_diagnostics(True)
for R, L in ((1, 256), (1, 1152), (32, 256), (32, 1152)):
    for qh, kvh in ((12, 4), (4, 4), (12, 12)):
        x = build(R, L, qh, kvh)
        ms = per_layer_ms(x, R, L, kvh)
        mx.eval(call(x, R, L, kvh))
        route = ops.last_paged_dispatch()
        kv_mb = R * L * kvh * HEAD * 2 * 2 / 1e6
        print(
            f"rows {R:2d} ctx {L:4d} heads {qh:2d}q/{kvh:2d}kv: {ms * 1e3:7.1f} us/layer, "
            f"K/V {kv_mb:5.1f} MB -> {kv_mb / ms:6.1f} GB/s on unique K/V  [{route}]",
            flush=True,
        )
        del x
ops._set_paged_dispatch_diagnostics(previous)
