"""Turn a sweep's step logs into achieved bandwidth and a roofline chart.

    .venv/bin/python bench/roofline.py results/runs/<date>-<label>

For each grid point it keeps the steady decode steps: no prompt tokens, as
many decode rows as the client concurrency, and the steps before and after
alike. The step period is the time from one step start to the next.

Bytes per step are the bytes the step must read (see "Achieved bandwidth" in
GLOSSARY.md), from the served checkpoint's tensor sizes:
  every step   attention, shared expert, router and norms of all layers
  output       the output layer, read once for all rows (the tied embedding)
  experts      distinct experts per layer (from the experts phase) x the
               bytes of one expert
  cache        the decode rows' context lengths x cache bytes per token,
               plus one new token per row
Activations are not counted, so the real traffic is a little higher.

FLOPs per step: 2 x the weights one token uses, per decode row, plus the
attention scores and sums (4 x heads x head_dim per context token per layer).

Writes roofline.json, summary.md and roofline.png into the run folder.
"""

import argparse
import json
import statistics
import struct
import sys
from pathlib import Path

from sweep import model_dir

ROOT = Path(__file__).resolve().parent.parent
PEAK_BANDWIDTH = 200e9
PEAK_FLOPS = 5.2e12
# Light surface and categorical slots 1-3 of the dataviz reference palette.
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def tensor_bytes(model: Path) -> dict[str, int]:
    sizes = {}
    for path in model.glob("*.safetensors"):
        with path.open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        header.pop("__metadata__", None)
        for name, info in header.items():
            start, end = info["data_offsets"]
            sizes[name] = end - start
    return sizes


def model_costs(model: Path) -> dict:
    config = json.loads((model / "config.json").read_text())
    sizes = tensor_bytes(model)
    layers, experts = config["num_hidden_layers"], config["num_local_experts"]
    d, f, fs = (
        config["hidden_size"],
        config["intermediate_size"],
        config["intermediate_size_mlp"],
    )
    heads, kv_heads, head_dim = (
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
    )

    expert = sum(v for k, v in sizes.items() if ".feed_forward.experts." in k)
    embed = sum(v for k, v in sizes.items() if k.startswith("model.embed_tokens."))
    lm_head = sum(v for k, v in sizes.items() if k.startswith("lm_head."))
    weights_per_token = (
        layers
        * (
            2 * d * heads * head_dim
            + 2 * d * kv_heads * head_dim
            + 3 * d * fs
            + d * experts
            + config["num_experts_per_tok"] * 3 * d * f
        )
        + config["vocab_size"] * d
    )
    return {
        "every_step_bytes": sum(sizes.values()) - expert - embed - lm_head,
        "output_bytes": lm_head or embed,
        "embedding_row_bytes": embed / config["vocab_size"],
        "expert_bytes": expert / (layers * experts),
        "cache_bytes_per_token": layers * 2 * kv_heads * head_dim * 2,
        "flops_per_token": 2 * weights_per_token,
        "flops_per_context_token": 4 * heads * head_dim * layers,
        "total_weight_bytes": sum(sizes.values()),
        "layers": layers,
    }


def load_steps(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def in_window(steps: list[dict], window: list[float]) -> list[dict]:
    return [s for s in steps if window[0] <= s["wall"] <= window[1]]


def is_steady(step: dict, rows: int) -> bool:
    return step["rows"] == rows and step["prefill"] == 0


def steady_steps(steps: list[dict], rows: int) -> list[dict]:
    """Steady steps of one run, each with the period to the next step."""
    out = []
    for prev, cur, nxt in zip(steps, steps[1:], steps[2:]):
        if all(is_steady(s, rows) for s in (prev, cur, nxt)):
            out.append({**cur, "period_ms": (nxt["t"] - cur["t"]) * 1e3})
    return out


def analyse(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    runs = json.loads((run_dir / "runs.json").read_text())
    costs = model_costs(model_dir(meta["model"]))
    timed = load_steps(run_dir / "steps-timed.jsonl")
    counted = load_steps(run_dir / "steps-experts.jsonl")

    points = []
    for input_len in meta["grid"]["input_lens"]:
        for c in meta["grid"]["concurrency"]:
            point_runs = [
                r for r in runs if r["input_len"] == input_len and r["concurrency"] == c
            ]
            steady, run_medians, clients, distinct = [], [], [], []
            warnings, swap_rates = [], []
            for r in point_runs:
                if r["phase"] == "experts":
                    distinct += [
                        sum(s["experts"]) / len(s["experts"])
                        for s in in_window(counted, r["window"])
                        if is_steady(s, c)
                    ]
                    continue
                steps = steady_steps(in_window(timed, r["window"]), c)
                steady += steps
                if steps:
                    run_medians.append(statistics.median(s["period_ms"] for s in steps))
                clients.append(
                    json.loads((run_dir / "client" / f"{r['name']}.json").read_text())
                )
                seconds = r["after"]["time"] - r["before"]["time"]
                swap_rates.append(
                    (r["after"]["swapouts"] - r["before"]["swapouts"]) / seconds
                )
                if r["after"]["thermal_warning"]:
                    warnings.append(f"{r['name']}: thermal warning")

            if not steady or not distinct:
                sys.exit(f"No steady steps for input {input_len}, concurrency {c}.")
            period = statistics.median(s["period_ms"] for s in steady) / 1e3
            context = statistics.mean(s["ctx"] for s in steady)
            experts_per_layer = statistics.mean(distinct)
            parts = {
                "every_step": costs["every_step_bytes"],
                "output": costs["output_bytes"],
                "experts": experts_per_layer * costs["layers"] * costs["expert_bytes"],
                "cache": (context + c) * costs["cache_bytes_per_token"],
                "embedding_rows": c * costs["embedding_row_bytes"],
            }
            total = sum(parts.values())
            flops = (
                c * costs["flops_per_token"]
                + context * costs["flops_per_context_token"]
            )
            points.append(
                {
                    "input_len": input_len,
                    "concurrency": c,
                    "steady_steps": len(steady),
                    "step_ms": period * 1e3,
                    "step_ms_p95": statistics.quantiles(
                        [s["period_ms"] for s in steady], n=20
                    )[-1],
                    "step_ms_run_medians": run_medians,
                    "build_ms": statistics.median(s["build_ms"] for s in steady),
                    "build_cpu_ms": statistics.median(
                        s["build_cpu_ms"] for s in steady
                    ),
                    "decode_tokens_per_s": c / period,
                    "context_tokens": context,
                    "distinct_experts_per_layer": experts_per_layer,
                    "bytes": parts,
                    "bytes_total": total,
                    "flops": flops,
                    "achieved_gb_per_s": total / period / 1e9,
                    "bandwidth_fraction": total / period / PEAK_BANDWIDTH,
                    "achieved_tflops": flops / period / 1e12,
                    "arithmetic_intensity": flops / total,
                    "min_step_ms": max(total / PEAK_BANDWIDTH, flops / PEAK_FLOPS)
                    * 1e3,
                    "client_output_tokens_per_s": statistics.median(
                        x["output_throughput"] for x in clients
                    ),
                    "client_median_itl_ms": statistics.median(
                        x["median_itl_ms"] for x in clients
                    ),
                    "client_median_ttft_ms": statistics.median(
                        x["median_ttft_ms"] for x in clients
                    ),
                    "swapouts_per_s": swap_rates,
                    "warnings": warnings,
                }
            )
    return {
        "run": run_dir.name,
        "idle_swapouts_per_s": meta.get("idle_swap_per_s", {}).get("swapouts"),
        "peak_bandwidth": PEAK_BANDWIDTH,
        "peak_flops": PEAK_FLOPS,
        "costs": costs,
        "points": points,
    }


def write_summary(result: dict, path: Path) -> None:
    lines = [
        f"# {result['run']}: roofline summary",
        "",
        "Steady decode steps only. Step time is the median; the range is the "
        "spread of the per-run medians. Generated by `bench/roofline.py`.",
        "",
        "| Input | Concurrency | Steps | Step ms (run range) | Submit call ms | "
        "Submit CPU ms | "
        "Decode tok/s | Distinct experts / layer | MB read / step | GB/s | "
        "% of peak | FLOP/byte | Min ms at peak |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in result["points"]:
        lo, hi = min(p["step_ms_run_medians"]), max(p["step_ms_run_medians"])
        lines.append(
            f"| {p['input_len']} | {p['concurrency']} | {p['steady_steps']} | "
            f"{p['step_ms']:.2f} ({lo:.2f}–{hi:.2f}) | {p['build_ms']:.2f} | "
            f"{p['build_cpu_ms']:.2f} | "
            f"{p['decode_tokens_per_s']:.0f} | "
            f"{p['distinct_experts_per_layer']:.1f} | "
            f"{p['bytes_total'] / 1e6:.0f} | {p['achieved_gb_per_s']:.1f} | "
            f"{100 * p['bandwidth_fraction']:.1f} | "
            f"{p['arithmetic_intensity']:.2f} | {p['min_step_ms']:.2f} |"
        )
    warnings = [w for p in result["points"] for w in p["warnings"]]
    rates = [r for p in result["points"] for r in p["swapouts_per_s"]]
    idle = result["idle_swapouts_per_s"]
    lines += [
        "",
        f"Swap-outs during runs: {min(rates):.0f}–{max(rates):.0f} pages/s "
        f"(idle before the sweep: "
        f"{'not measured' if idle is None else f'{idle:.0f} pages/s'}).",
        "",
        "Other machine-state warnings: " + ("; ".join(warnings) or "none"),
        "",
    ]
    path.write_text("\n".join(lines))


def plot(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = result["points"]
    lens = sorted({p["input_len"] for p in points})
    plt.rcParams.update(
        {
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(
        1, 1 + len(lens), figsize=(4.2 * (1 + len(lens)), 3.8), facecolor=SURFACE
    )
    roof, *steps = axes
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color="#e4e3df", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    lo = min(p["arithmetic_intensity"] for p in points) / 2
    hi = max(100.0, max(p["arithmetic_intensity"] for p in points) * 2)
    xs = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    roof.plot(
        xs,
        [min(PEAK_FLOPS, x * PEAK_BANDWIDTH) / 1e12 for x in xs],
        color=MUTED,
        linewidth=2,
    )
    # Right-anchored, so the text stays above the rising bandwidth line.
    roof.text(
        lo * 6,
        lo * 6.6 * PEAK_BANDWIDTH / 1e12,
        "200 GB/s",
        color=MUTED,
        ha="right",
        va="bottom",
    )
    roof.text(hi / 1.1, PEAK_FLOPS / 1e12 * 1.15, "5.2 TFLOPS", color=MUTED, ha="right")
    # Labels go below-right for the first series, above-left for the second.
    offsets = [(5, -11), (-12, 5)]
    for color, offset, n in zip(SERIES, offsets, lens):
        series = [p for p in points if p["input_len"] == n]
        x = [p["arithmetic_intensity"] for p in series]
        y = [p["achieved_tflops"] for p in series]
        roof.plot(
            x,
            y,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=f"{n}-token prompts",
        )
        for p in series:
            roof.annotate(
                str(p["concurrency"]),
                (p["arithmetic_intensity"], p["achieved_tflops"]),
                textcoords="offset points",
                xytext=offset,
                color=INK,
                fontsize=8,
            )
    roof.set(
        xscale="log",
        yscale="log",
        xlabel="Arithmetic intensity (FLOP per byte read)",
        ylabel="Achieved TFLOP/s",
        title="Roofline (labels: concurrency)",
    )
    roof.legend(frameon=False, loc="upper left")

    for ax, n in zip(steps, lens):
        series = [p for p in points if p["input_len"] == n]
        rows = [p["concurrency"] for p in series]
        for key, label, color in (
            ("step_ms", "Measured step", SERIES[0]),
            ("build_ms", "Submit call, wall time", SERIES[1]),
            ("build_cpu_ms", "Submit call, CPU time", SERIES[2]),
            ("min_step_ms", "Roofline lower bound", MUTED),
        ):
            ax.plot(
                rows,
                [p[key] for p in series],
                "o-",
                color=color,
                linewidth=2,
                markersize=5,
                label=label,
            )
        ax.set(
            xscale="log",
            xlabel="Decode rows per step",
            ylabel="Milliseconds per step",
            title=f"Where a step's time goes ({n}-token prompts)",
            ylim=(0, None),
        )
        ax.set_xticks(rows, [str(r) for r in rows])
        ax.minorticks_off()
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = analyse(args.run_dir)
    (args.run_dir / "roofline.json").write_text(json.dumps(result, indent=2) + "\n")
    write_summary(result, args.run_dir / "summary.md")
    plot(result, args.run_dir / "roofline.png")
    print((args.run_dir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
