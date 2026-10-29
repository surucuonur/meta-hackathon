"""Run the benchmark sweep against a real `vllm serve` process.

    .venv/bin/python bench/sweep.py --label baseline

Needs the step-log plugin installed in the same environment
(uv pip install -e bench/steplog). Two phases, each with its own server:

  timed    Every grid point (client concurrency x input length) runs --reps
           times. Each repetition uses a new shuffled order, so slow drift
           (heat, background load) does not line up with one setting.
  experts  Every grid point once more, with distinct-expert counting on.
           Counting reads the routing results back from the GPU in every
           step, so this phase gives expert counts, not timings.

In the timed phase, each client run first sends one wave of warmup requests
(one per in-flight slot). The same seed gives the same prompts in both
phases. Results go to results/runs/<date>-<label>/:

  meta.json             machine, versions, commits, settings
  runs.json             one entry per client run: time window, machine state
  steps-<phase>.jsonl   engine step log (see bench/steplog)
  client/               vllm bench serve result and output per run
  server-<phase>.log
"""

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import date
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = Path(sys.executable).parent
PORT = 8000
SEED = 1
OUTPUT_LEN = 256
# A fixed attention cache size, so the cache does not depend on the memory
# that is free at startup. It is checked against the grid at startup.
KV_CACHE_BYTES = 1 << 30


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def git_commit(repo: Path) -> str:
    sha = sh("git", "-C", str(repo), "rev-parse", "HEAD").strip()
    dirty = sh("git", "-C", str(repo), "status", "--porcelain").strip()
    return sha + ("-dirty" if dirty else "")


def model_dir(model: str) -> Path:
    """A local model folder, or a Hugging Face model id in the local cache."""
    if Path(model).is_dir():
        return Path(model).resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model,
            allow_patterns=["*.json", "*.py", "*.safetensors"],
            local_files_only=True,
        )
    )


def as_args(flags: dict) -> list[str]:
    args = []
    for flag, value in flags.items():
        args += [flag] if value is None else [flag, str(value)]
    return args


def machine_info() -> dict:
    gpu = json.loads(sh("system_profiler", "SPDisplaysDataType", "-json"))
    return {
        "model": sh("sysctl", "-n", "hw.model").strip(),
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string").strip(),
        "gpu_cores": int(gpu["SPDisplaysDataType"][0]["sppci_cores"]),
        "memory_gib": int(sh("sysctl", "-n", "hw.memsize")) / 2**30,
        "macos": sh("sw_vers", "-productVersion").strip(),
        "macos_build": sh("sw_vers", "-buildVersion").strip(),
        "power_source": sh("pmset", "-g", "batt").splitlines()[0],
        "low_power_mode": bool(re.search(r"lowpowermode\s+1", sh("pmset", "-g"))),
    }


def gpu_utilization() -> int | None:
    """Device-wide GPU utilization in percent, or None if macOS doesn't report it."""
    out = sh("ioreg", "-r", "-c", "AGXAccelerator", "-d", "1")
    match = re.search(r'"Device Utilization %"=(\d+)', out)
    return int(match.group(1)) if match else None


def machine_state() -> dict:
    swap = re.search(r"used = ([\d.]+)M", sh("sysctl", "-n", "vm.swapusage"))
    free = re.search(r"percentage: (\d+)%", sh("memory_pressure"))
    counters = re.findall(r"^(Swapins|Swapouts|Pageouts):\s+(\d+)", sh("vm_stat"), re.M)
    therm = sh("pmset", "-g", "therm")
    return {
        "time": time.time(),
        "swap_used_mb": float(swap.group(1)),
        "memory_free_percent": int(free.group(1)),
        "thermal_warning": "No thermal warning level has been recorded" not in therm,
        **{name.lower(): int(count) for name, count in counters},
    }


def idle_swap_rate(seconds: float = 30) -> dict:
    """Swap activity per second while no benchmark runs."""
    before = machine_state()
    time.sleep(seconds)
    after = machine_state()
    elapsed = after["time"] - before["time"]
    keys = ("swapins", "swapouts", "pageouts")
    return {k: (after[k] - before[k]) / elapsed for k in keys}


class GpuSampler(threading.Thread):
    """Samples device-wide GPU utilization once per second."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples: list[int | None] = []
        self.done = threading.Event()

    def run(self):
        while not self.done.wait(1.0):
            self.samples.append(gpu_utilization())

    def stop(self) -> list[int | None]:
        self.done.set()
        self.join()
        return self.samples


def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
            return True
    except OSError:
        return False


class Server:
    def __init__(self, path: Path, args: list[str], env: dict, log: Path):
        self.log = log
        self.proc = subprocess.Popen(
            [str(BIN / "vllm"), "serve", str(path), *args],
            env={**os.environ, **env},
            stdout=log.open("w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(self, timeout: float = 600) -> None:
        end = time.time() + timeout
        while not server_up():
            if self.proc.poll() is not None:
                tail = "\n".join(self.log.read_text().splitlines()[-20:])
                sys.exit(f"Server exited during startup:\n{tail}")
            if time.time() > end:
                sys.exit(f"Server not ready after {timeout} s; see {self.log}")
            time.sleep(1)

    def kv_cache_tokens(self) -> int:
        match = re.search(r"KV cache size: ([\d,]+) tokens", self.log.read_text())
        return int(match.group(1).replace(",", ""))

    def stop(self) -> None:
        if self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGINT)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()


def run_client(path: Path, out: Path, phase: str, rep: int, input_len: int, c: int):
    name = f"{phase}-in{input_len}-c{c}-r{rep}"
    flags = {
        "--backend": "openai",
        "--base-url": f"http://127.0.0.1:{PORT}",
        "--endpoint": "/v1/completions",
        "--model": path,
        "--served-model-name": "bench",
        "--trust-remote-code": None,
        "--dataset-name": "random",
        "--random-input-len": input_len,
        "--random-output-len": OUTPUT_LEN,
        "--random-range-ratio": 0,
        "--num-prompts": max(8, 4 * c),
        "--num-warmups": c if phase == "timed" else 0,
        "--request-rate": "inf",
        "--max-concurrency": c,
        "--ignore-eos": None,
        "--temperature": 0,
        "--seed": SEED,
        "--percentile-metrics": "ttft,tpot,itl,e2el",
        "--metric-percentiles": "50,95,99",
        "--ready-check-timeout-sec": 0,
        "--disable-tqdm": None,
        "--save-result": None,
        "--save-detailed": None,
        "--result-dir": out / "client",
        "--result-filename": f"{name}.json",
    }
    idle_gpu = []
    for _ in range(3):
        idle_gpu.append(gpu_utilization())
        time.sleep(0.5)
    before = machine_state()
    sampler = GpuSampler()
    sampler.start()

    # The measured window starts when the client finishes its warmup wave.
    started = None
    with (out / "client" / f"{name}.log").open("w") as log:
        proc = subprocess.Popen(
            [str(BIN / "vllm"), "bench", "serve", *as_args(flags)],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            if started is None and line.startswith("Starting main benchmark run"):
                started = time.time()
            log.write(line)
        proc.wait()
    ended = time.time()
    gpu = sampler.stop()
    after = machine_state()
    if proc.returncode != 0 or started is None:
        sys.exit(f"Client run {name} failed; see {out / 'client' / name}.log")

    result = json.loads((out / "client" / f"{name}.json").read_text())
    print(
        f"{name:<24} {result['output_throughput']:8.1f} tok/s  "
        f"median ITL {result['median_itl_ms']:6.2f} ms  "
        f"swapouts +{after['swapouts'] - before['swapouts']}",
        flush=True,
    )
    return {
        "name": name,
        "phase": phase,
        "rep": rep,
        "input_len": input_len,
        "concurrency": c,
        "window": [started, ended],
        "idle_gpu_percent": idle_gpu,
        "gpu_percent": gpu,
        "before": before,
        "after": after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--model",
        default="models/MobileMoE-S-QAT-mlx-bfloat16",
        help="model folder or Hugging Face id (default: bench/convert_qat.py output)",
    )
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--input-lens", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["timed", "experts"],
        default=["timed", "experts"],
    )
    args = parser.parse_args()

    out = ROOT / "results" / "runs" / f"{date.today()}-{args.label}"
    if out.exists():
        sys.exit(f"{out} exists; choose another --label")
    if server_up():
        sys.exit(f"Something is already serving on port {PORT}; stop it first.")
    path = model_dir(args.model)
    (out / "client").mkdir(parents=True)
    grid = [(n, c) for n in args.input_lens for c in args.concurrency]
    kv_tokens_needed = max(args.concurrency) * (max(args.input_lens) + OUTPUT_LEN)
    server_flags = {
        "--host": "127.0.0.1",
        "--port": PORT,
        "--served-model-name": "bench",
        "--trust-remote-code": None,
        "--max-model-len": 2048,
        "--max-num-seqs": max(args.concurrency),
        "--max-num-batched-tokens": 2048,
        "--kv-cache-memory-bytes": KV_CACHE_BYTES,
        "--no-enable-prefix-caching": None,
        "--generation-config": "vllm",
    }
    meta = {
        "label": args.label,
        "date": str(date.today()),
        "machine": machine_info(),
        "machine_state_at_start": machine_state(),
        "idle_swap_per_s": idle_swap_rate(),
        "versions": {p: version(p) for p in ["vllm", "vllm-metal", "mlx", "mlx-lm"]},
        "commits": {
            "repo": git_commit(ROOT),
            **{
                name: git_commit(ROOT / "third_party" / name)
                for name in ["vllm-metal", "mlx-lm", "vllm"]
            },
        },
        "model": args.model,
        "revision": path.name,
        "converted_from": json.loads((path / "config.json").read_text()).get(
            "converted_from"
        ),
        "server_env": {"VLLM_ENABLE_V1_MULTIPROCESSING": "1"},
        "server_flags": server_flags,
        "client": {
            "dataset": "random",
            "output_len": OUTPUT_LEN,
            "seed": SEED,
            "num_prompts": "max(8, 4 x concurrency)",
            "warmups": "concurrency in the timed phase, 0 in the experts phase",
        },
        "grid": {"input_lens": args.input_lens, "concurrency": args.concurrency},
        "reps": args.reps,
        "kv_cache_tokens": {"needed": kv_tokens_needed},
    }

    runs = []
    for phase in args.phases:
        env = {
            "HF_HUB_OFFLINE": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
            "BENCH_STEP_LOG": str(out / f"steps-{phase}.jsonl"),
        }
        if phase == "experts":
            env["BENCH_EXPERT_LOG"] = "1"
        server = Server(path, as_args(server_flags), env, out / f"server-{phase}.log")
        try:
            server.wait_ready()
            meta["kv_cache_tokens"][phase] = server.kv_cache_tokens()
            (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            if meta["kv_cache_tokens"][phase] < kv_tokens_needed:
                sys.exit(f"KV cache too small for the grid; see {out / 'meta.json'}")
            reps = args.reps if phase == "timed" else 1
            for rep in range(reps):
                order = grid[:]
                random.Random(SEED + rep).shuffle(order)
                for input_len, c in order:
                    runs.append(run_client(path, out, phase, rep, input_len, c))
                    (out / "runs.json").write_text(json.dumps(runs, indent=1) + "\n")
        finally:
            server.stop()
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
