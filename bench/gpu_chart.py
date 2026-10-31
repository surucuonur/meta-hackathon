"""Chart GPU utilization over time for every client run of a sweep.

    .venv/bin/python bench/gpu_chart.py results/runs/<date>-<label>

One panel per grid point: the timed repeats and the expert-counting run, in
seconds from the start of the measured run. Negative times are the client
starting up and sending its warmup requests. Writes gpu-utilization.png into
the run folder.
"""

import argparse
import json
from pathlib import Path

from roofline import INK, MUTED, SERIES, SURFACE
from run_tables import gpu_timeline

# Neutral midpoint of the dataviz reference palette, for the measured-run band.
BAND = "#f0efec"


def plot(run_dir: Path, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    meta = json.loads((run_dir / "meta.json").read_text())
    runs = json.loads((run_dir / "runs.json").read_text())
    lens, conc = meta["grid"]["input_lens"], meta["grid"]["concurrency"]
    plt.rcParams.update(
        {
            "font.size": 8,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(
        len(lens),
        len(conc),
        figsize=(2.6 * len(conc), 2.5 * len(lens) + 0.9),
        sharey=True,
        facecolor=SURFACE,
        squeeze=False,
    )
    for row, input_len in enumerate(lens):
        for col, c in enumerate(conc):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            ax.grid(color="#e4e3df", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            point = [
                r for r in runs if (r["input_len"], r["concurrency"]) == (input_len, c)
            ]
            timed = sorted(
                (r for r in point if r["phase"] == "timed"), key=lambda r: r["rep"]
            )
            ax.axvspan(
                0, max(r["window"][1] - r["window"][0] for r in timed), color=BAND
            )
            for r in point:
                t, busy = zip(*gpu_timeline(r))
                if r["phase"] == "timed":
                    ax.plot(t, busy, color=SERIES[r["rep"]], linewidth=1.4)
                else:
                    ax.plot(t, busy, color=MUTED, linewidth=1.1, linestyle="--")
            ax.set_ylim(-4, 104)
            ax.set_yticks([0, 50, 100])
            if row == 0:
                ax.set_title(f"{c} request{'s' if c > 1 else ''}", color=INK)
            if col == 0:
                ax.set_ylabel(f"{input_len}-token prompts\nGPU busy %")
            if row == len(lens) - 1:
                ax.set_xlabel("Seconds from start of measured run")

    handles = [
        Line2D([], [], color=color, linewidth=1.4, label=f"Timed run {i + 1}")
        for i, color in enumerate(SERIES[: meta["reps"]])
    ]
    handles += [
        Line2D(
            [],
            [],
            color=MUTED,
            linewidth=1.1,
            linestyle="--",
            label="Expert-counting run",
        ),
        Patch(color=BAND, label="Measured run (timed runs)"),
    ]
    fig.suptitle(
        "GPU utilization during each client run "
        "(macOS device-wide counter, about one reading per second)",
        y=0.985,
        fontsize=10,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=160, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    run_dir = parser.parse_args().run_dir.resolve()
    plot(run_dir, run_dir / "gpu-utilization.png")
    print(run_dir / "gpu-utilization.png")


if __name__ == "__main__":
    main()
