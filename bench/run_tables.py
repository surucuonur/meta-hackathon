"""Print a sweep's per-run and per-point tables as Markdown.

    .venv/bin/python bench/run_tables.py results/runs/<date>-<label>

Run bench/roofline.py first; this reads its roofline.json. The tables are the
ones in documents/benchmark/measurements.md.
"""

import argparse
import json
import statistics
from pathlib import Path

from roofline import PEAK_BANDWIDTH, in_window, load_steps, steady_steps


def gpu_timeline(run: dict) -> list[tuple[float, int]]:
    """GPU samples as (seconds from the measured window's start, percent).

    The samples carry no timestamps. The sampler starts right after
    run["before"] and stops when the client exits, taking one sample per second
    plus the time ioreg needs, so the samples are spread evenly over that span.
    """
    samples = run["gpu_percent"]
    start, end = run["before"]["time"], run["window"][1]
    spacing = (end - start) / len(samples)
    return [
        (start + (i + 1) * spacing - run["window"][0], x)
        for i, x in enumerate(samples)
        if x is not None
    ]


def window_gpu(run: dict) -> list[int]:
    """GPU samples taken during the measured window."""
    length = run["window"][1] - run["window"][0]
    return [x for t, x in gpu_timeline(run) if 0 <= t <= length]


def swap_rate(run: dict) -> float:
    seconds = run["after"]["time"] - run["before"]["time"]
    return (run["after"]["swapouts"] - run["before"]["swapouts"]) / seconds


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append(
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"
    )
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def per_run(run_dir: Path, runs: list[dict], steps: dict) -> str:
    order = {r["name"]: i + 1 for i, r in enumerate(runs)}
    rows = []
    for r in sorted(
        runs,
        key=lambda r: (
            r["phase"] != "timed",
            r["input_len"],
            r["concurrency"],
            r["rep"],
        ),
    ):
        client = json.loads((run_dir / "client" / f"{r['name']}.json").read_text())
        st = steady_steps(in_window(steps[r["phase"]], r["window"]), r["concurrency"])
        periods = [s["period_ms"] for s in st]
        gpu = window_gpu(r)
        rows.append(
            [
                r["name"],
                order[r["name"]],
                f"{client['output_throughput']:.1f}",
                f"{client['median_itl_ms']:.2f}",
                f"{client['p99_itl_ms']:.2f}",
                f"{client['median_ttft_ms']:.0f}",
                len(st),
                f"{statistics.median(periods):.2f}",
                f"{statistics.quantiles(periods, n=20)[-1]:.2f}",
                f"{statistics.median(s['build_cpu_ms'] for s in st):.2f}",
                f"{statistics.mean(gpu):.0f} / {statistics.median(gpu):.0f}",
                f"{swap_rate(r):.0f}",
                r["before"]["memory_free_percent"],
            ]
        )
    return table(
        [
            "Run",
            "Order",
            "Client tok/s",
            "Median ITL ms",
            "p99 ITL ms",
            "Median TTFT ms",
            "Steady steps",
            "Step ms",
            "Step ms p95",
            "Submit CPU ms",
            "GPU busy % (mean / median)",
            "Swap-outs/s",
            "Free memory % before",
        ],
        rows,
    )


def bytes_and_time(points: list[dict]) -> str:
    rows = []
    for p in points:
        b, total = p["bytes"], p["bytes_total"]
        rows.append(
            [
                p["input_len"],
                p["concurrency"],
                f"{b['every_step'] / 1e6:.1f}",
                f"{b['output'] / 1e6:.1f}",
                f"{b['experts'] / 1e6:.1f}",
                f"{b['cache'] / 1e6:.1f}",
                f"{total / 1e6:.1f}",
                f"{p['step_ms']:.2f}",
                f"{p['build_ms']:.2f}",
                f"{p['build_cpu_ms']:.2f}",
                f"{p['build_ms'] - p['build_cpu_ms']:.2f}",
                f"{p['min_step_ms']:.2f}",
                f"{p['step_ms'] / p['min_step_ms']:.1f}",
            ]
        )
    return table(
        [
            "Prompt",
            "Requests",
            "Every-step weights MB",
            "Output layer MB",
            "Experts MB",
            "Cache MB",
            "Total MB",
            "Step ms",
            "Submit call ms",
            "Submit CPU ms",
            "Submit waiting ms",
            "Lower bound ms",
            "Step / lower bound",
        ],
        rows,
    )


def client_view(points: list[dict]) -> str:
    rows = [
        [
            p["input_len"],
            p["concurrency"],
            f"{p['client_output_tokens_per_s']:.0f}",
            f"{p['decode_tokens_per_s']:.0f}",
            f"{p['client_median_itl_ms']:.2f}",
            f"{p['client_median_ttft_ms']:.0f}",
            f"{p['step_ms_p95']:.2f}",
            f"{p['achieved_tflops']:.2f}",
            f"{p['arithmetic_intensity']:.1f}",
        ]
        for p in points
    ]
    return table(
        [
            "Prompt",
            "Requests",
            "Client tok/s",
            "Engine decode tok/s",
            "Median ITL ms",
            "Median TTFT ms",
            "Step ms p95",
            "Achieved TFLOPS",
            "FLOP/byte",
        ],
        rows,
    )


def added_request(points: list[dict]) -> str:
    """What each request added between neighbouring grid points costs."""
    rows = []
    for a, b in zip(points, points[1:]):
        if a["input_len"] != b["input_len"]:
            continue
        added = b["concurrency"] - a["concurrency"]
        ms = (b["step_ms"] - a["step_ms"]) / added
        mb = (b["bytes_total"] - a["bytes_total"]) / added / 1e6
        at_peak = mb * 1e6 / PEAK_BANDWIDTH * 1e3
        rows.append(
            [
                a["input_len"],
                f"{a['concurrency']} to {b['concurrency']}",
                f"{ms:.2f}",
                f"{(b['build_cpu_ms'] - a['build_cpu_ms']) / added:.2f}",
                f"{mb:.1f}",
                f"{at_peak:.2f}",
                f"{ms / at_peak:.1f}",
            ]
        )
    return table(
        [
            "Prompt",
            "Requests",
            "Step ms per added request",
            "Submit CPU ms per added request",
            "MB per added request",
            "Ms at peak bandwidth",
            "Ratio",
        ],
        rows,
    )


def repeats(points: list[dict]) -> str:
    """How far each repeat's median step time is from the setting's median."""
    rows = []
    for p in points:
        medians = p["step_ms_run_medians"]
        worst = max(abs(m - p["step_ms"]) for m in medians) / p["step_ms"]
        rows.append(
            [
                p["input_len"],
                p["concurrency"],
                f"{p['step_ms']:.2f}",
                ", ".join(f"{m:.2f}" for m in medians),
                f"{100 * worst:.1f}",
                ", ".join(f"{x:.0f}" for x in p["swapouts_per_s"]),
            ]
        )
    return table(
        [
            "Prompt",
            "Requests",
            "Step ms (all steady steps)",
            "Step ms per repeat (r0, r1, r2)",
            "Largest gap %",
            "Swap-outs/s per repeat",
        ],
        rows,
    )


def counting_cost(points: list[dict], runs: list[dict], counted: list[dict]) -> str:
    rows = []
    for p in points:
        run = next(
            r
            for r in runs
            if r["phase"] == "experts"
            and (r["input_len"], r["concurrency"]) == (p["input_len"], p["concurrency"])
        )
        st = steady_steps(in_window(counted, run["window"]), p["concurrency"])
        step = statistics.median(s["period_ms"] for s in st)
        per_layer = [x for s in st for x in s["experts"]]
        rows.append(
            [
                p["input_len"],
                p["concurrency"],
                f"{p['step_ms']:.2f}",
                f"{step:.2f}",
                f"+{step - p['step_ms']:.2f}",
                f"{statistics.mean(per_layer):.1f}",
                min(per_layer),
                max(per_layer),
            ]
        )
    return table(
        [
            "Prompt",
            "Requests",
            "Step ms (timed)",
            "Step ms (counting)",
            "Added",
            "Distinct experts / layer, mean",
            "Min",
            "Max",
        ],
        rows,
    )


def prefill(runs: list[dict], timed: list[dict], flops_per_token: float) -> str:
    """Steps with at least 1,000 prompt tokens: their period and weight FLOP rate."""
    groups = {}
    for r in runs:
        if r["phase"] != "timed":
            continue
        steps = in_window(timed, r["window"])
        for cur, nxt in zip(steps, steps[1:]):
            if cur["prefill"] >= 1000:
                kind = "with decode rows" if cur["rows"] else "prompt only"
                groups.setdefault((r["input_len"], kind), []).append(
                    (cur, nxt["t"] - cur["t"])
                )
    rows = []
    for (input_len, kind), steps in sorted(groups.items()):
        tflops = [s["prefill"] * flops_per_token / period / 1e12 for s, period in steps]
        rows.append(
            [
                input_len,
                kind,
                len(steps),
                statistics.median(s["prefill"] for s, _ in steps),
                statistics.median(s["rows"] for s, _ in steps),
                f"{statistics.median(period for _, period in steps) * 1e3:.0f}",
                f"{statistics.median(s['build_cpu_ms'] for s, _ in steps):.0f}",
                f"{statistics.median(tflops):.2f}",
            ]
        )
    return table(
        [
            "Prompt",
            "Step",
            "Steps",
            "Prompt tokens (median)",
            "Decode rows (median)",
            "Step ms",
            "Submit CPU ms",
            "Prompt TFLOPS (weights only)",
        ],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    run_dir = parser.parse_args().run_dir.resolve()
    runs = json.loads((run_dir / "runs.json").read_text())
    roof = json.loads((run_dir / "roofline.json").read_text())
    steps = {
        "timed": load_steps(run_dir / "steps-timed.jsonl"),
        "experts": load_steps(run_dir / "steps-experts.jsonl"),
    }
    points = roof["points"]
    timed_gpu = [x for r in runs if r["phase"] == "timed" for x in window_gpu(r)]

    print("## Bytes and time per decode step\n")
    print(bytes_and_time(points))
    print("\n## Client view and compute rate\n")
    print(client_view(points))
    print("\n## Cost of each added request\n")
    print(added_request(points))
    print("\n## Cost of counting experts\n")
    print(counting_cost(points, runs, steps["experts"]))
    print("\n## Steps with prompt tokens\n")
    print(prefill(runs, steps["timed"], roof["costs"]["flops_per_token"]))
    print("\n## Repeats\n")
    print(repeats(points))
    print("\n## Every run\n")
    print(per_run(run_dir, runs, steps))
    print(
        f"\nGPU samples in the timed windows: {len(timed_gpu)}, "
        f"{sum(x < 50 for x in timed_gpu)} below 50%."
    )


if __name__ == "__main__":
    main()
