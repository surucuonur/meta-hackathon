"""Investigation microbenchmarks, part 2: MoE block components over 20 layers.

Each measurement chains the component through all 20 layers (each layer's own
weights), so every kernel depends on the previous one as in a real step.
"""

import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
from mlx_lm.utils import load_model

MODEL = Path("models/MobileMoE-S-QAT-mlx-bfloat16")
model, config = load_model(MODEL)
mx.eval(model.parameters())
layers = model.model.layers
D = config["hidden_size"]
EXPERT_BYTES = 552_960


def med_ms(fn, reps=21, warm=4):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def rms(x):
    return mx.fast.rms_norm(x, None, 1e-5)


def routing(moe, x):
    scores = mx.sigmoid(moe.router(x.astype(mx.float32)))
    inds = mx.argpartition(-(scores + moe.expert_bias), kth=3, axis=-1)[..., :4]
    w = mx.take_along_axis(scores, inds, axis=-1)
    w = w / (w.sum(axis=-1, keepdims=True) + 1e-20) * moe.routed_scaling_factor
    return inds, w


# Transposed (standard MLX) copies of every layer's experts, re-quantized along
# the input dim. Numerics differ slightly; this is for timing only.
def transposed(proj):
    w = mx.dequantize(proj["weight"], proj["scales"], proj["biases"], 32, 4)
    return mx.quantize(mx.swapaxes(w, -1, -2), 32, 4)


T = []
for layer in layers:
    e = layer.feed_forward.experts
    T.append(
        {n: transposed(getattr(e, n)) for n in ("gate_proj", "up_proj", "down_proj")}
    )
mx.eval(T)


def experts_t(x, inds, t):
    """SwitchGLU with the transposed layout (same sorting rule as mlx_lm)."""
    x = mx.expand_dims(x, (-2, -3))
    do_sort = inds.size >= 64
    idx, inv = inds, None
    if do_sort:
        x, idx, inv = _gather_sort(x, inds)

    def mm(h, name):
        w, s, b = t[name]
        return mx.gather_qmm(
            h,
            w,
            s,
            b,
            rhs_indices=idx,
            transpose=True,
            group_size=32,
            bits=4,
            sorted_indices=do_sort,
        )

    g = mm(x, "gate_proj")
    h = (g * mx.sigmoid(g)) * mm(x, "up_proj")
    y = mm(h, "down_proj")
    if do_sort:
        y = _scatter_unsort(y, inv, inds.shape)
    return y.squeeze(-2)


for R in (1, 4, 16, 32):
    x0 = mx.random.normal((1, R, D)).astype(mx.bfloat16)
    # Real routing for this input at every layer (fixed for the expert-only runs).
    routes = [routing(l.feed_forward, rms(x0)) for l in layers]
    mx.eval(x0, routes)
    distinct = statistics.mean(len(set(r[0].reshape(-1).tolist())) for r in routes)

    def moe_stack(x):
        for l in layers:
            x = l.feed_forward(rms(x))
        return x

    compiled = mx.compile(moe_stack)

    def experts_stack(x):
        for l, (inds, w) in zip(layers, routes):
            y = l.feed_forward.experts(rms(x), inds)
            x = (y * w[..., None].astype(y.dtype)).sum(axis=-2)
        return x

    def experts_t_stack(x):
        for t, (inds, w) in zip(T, routes):
            y = experts_t(rms(x), inds, t)
            x = (y * w[..., None].astype(y.dtype)).sum(axis=-2)
        return x

    def shared_stack(x):
        for l in layers:
            x = l.feed_forward.shared_expert(rms(x))
        return x

    def attn_proj_stack(x):
        side = []
        for l in layers:
            a = l.self_attn
            h = rms(x)
            side += [a.k_proj(h), a.v_proj(h)]
            x = a.o_proj(a.q_proj(h))
        return x, side

    res = {
        "moe": med_ms(lambda: mx.eval(moe_stack(x0))),
        "moe_compiled": med_ms(lambda: mx.eval(compiled(x0))),
        "experts": med_ms(lambda: mx.eval(experts_stack(x0))),
        "experts_transposed": med_ms(lambda: mx.eval(experts_t_stack(x0))),
        "shared": med_ms(lambda: mx.eval(shared_stack(x0))),
        "attn_proj": med_ms(lambda: mx.eval(attn_proj_stack(x0))),
    }
    min_expert_ms = distinct * 20 * EXPERT_BYTES / 200e9 * 1e3
    pair_bytes = R * 4 * 20 * EXPERT_BYTES
    print(
        f"R={R:2d} distinct/layer {distinct:4.1f} | "
        + " | ".join(f"{k} {v:.2f}" for k, v in res.items())
        + f" | experts at peak {min_expert_ms:.2f} ms; per-pair reads {pair_bytes / 1e6:.0f} MB",
        flush=True,
    )

# Kernel choice at equal data: per-pair qvm vs the expert-grouped kernel
# (gather_qmm_rhs), which MLX picks only when sorted and pairs/experts >= 4.
print("kernel A/B (gate+up+down of one layer, x20 calls chained):", flush=True)
e0 = layers[0].feed_forward.experts
for R, E in ((8, 8), (16, 16), (32, 30)):
    B = R * 4
    inds = mx.sort(
        mx.array([[(i * 7 + k * 3) % E for k in range(4)] for i in range(R)]).reshape(
            -1
        )
    )
    x = mx.random.normal((B, 1, D)).astype(mx.bfloat16)  # one row per pair
    mx.eval(inds, x)
    for layout in ("ours", "transposed"):
        for sort in (False, True):
            if layout == "ours":
                ws = {
                    n: (
                        getattr(e0, n)["weight"][:E],
                        getattr(e0, n)["scales"][:E],
                        getattr(e0, n)["biases"][:E],
                    )
                    for n in ("gate_proj", "up_proj", "down_proj")
                }
                tr = False
            else:
                ws = {
                    n: tuple(a[:E] for a in T[0][n])
                    for n in ("gate_proj", "up_proj", "down_proj")
                }
                tr = True

            def run(h, ws=ws, tr=tr, sort=sort):
                for _ in range(20):
                    g = mx.gather_qmm(
                        h,
                        *ws["gate_proj"],
                        rhs_indices=inds,
                        transpose=tr,
                        group_size=32,
                        bits=4,
                        sorted_indices=sort,
                    )
                    u = mx.gather_qmm(
                        h,
                        *ws["up_proj"],
                        rhs_indices=inds,
                        transpose=tr,
                        group_size=32,
                        bits=4,
                        sorted_indices=sort,
                    )
                    h = mx.gather_qmm(
                        (g * mx.sigmoid(g)) * u,
                        *ws["down_proj"],
                        rhs_indices=inds,
                        transpose=tr,
                        group_size=32,
                        bits=4,
                        sorted_indices=sort,
                    )
                    h = mx.fast.rms_norm(h, None, 1e-5)
                return h

            ms = med_ms(lambda: mx.eval(run(x)))
            print(
                f"  rows {R:2d}, pairs {B:3d}, experts {E:2d}, {layout:10s} sorted={sort!s:5s}: "
                f"{ms:.2f} ms per 20 layers",
                flush=True,
            )
