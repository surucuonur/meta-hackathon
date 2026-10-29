"""Per-step records from vllm-metal's model runner, for the bandwidth study.

vLLM loads this module in every process as a general plugin. It does nothing
unless BENCH_STEP_LOG names an output file. Then each step that runs the
model appends one JSON line:

  wall      time.time() at the step start, to match the client's run windows
  t         time.perf_counter() at the step start; the interval to the next
            step's t is the step period
  build_ms  wall time of the call that builds and submits the step's GPU work
  build_cpu_ms  CPU time of the engine thread during that call; if it is
            well below build_ms, the call waited (for example on the GPU)
  rows      decode rows: requests advanced by one token
  prefill   prompt tokens processed in the same step
  ctx       sum of the decode rows' context lengths, in tokens: the cached
            attention state the step reads

With BENCH_EXPERT_LOG=1, each record with decode rows also has `experts`:
the number of distinct experts the decode rows used in each MoE layer.
Counting reads the routing results back from the GPU during every step, so
step times from that run are not measurements.

The hook wraps MetalModelRunner._start_paged_forward, which builds and
submits one step. It assumes no speculative decoding: one token per decode
row, decode rows first in the packed batch (as vllm-metal packs them).
"""

import importlib.abc
import importlib.util
import json
import os
import sys
import time

TARGET = "vllm_metal.v1.model_runner"


def register() -> None:
    path = os.environ.get("BENCH_STEP_LOG")
    if not path:
        return
    if TARGET in sys.modules:
        _patch(sys.modules[TARGET], path)
    elif not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        # Importing the runner here would load MLX into every vLLM process,
        # so patch it when the engine process first imports it.
        sys.meta_path.insert(0, _PatchOnImport(path))


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def __init__(self, path: str):
        self.path = path

    def find_spec(self, name, import_path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module, self.path)

        spec.loader.exec_module = exec_and_patch
        return spec


def _patch(module, path: str) -> None:
    runner = module.MetalModelRunner
    start_forward = runner._start_paged_forward
    count_experts = os.environ.get("BENCH_EXPERT_LOG") == "1"
    out = open(path, "a", buffering=1)
    routing = []

    if count_experts:
        import mlx.core as mx
        import numpy as np
        from mlx_lm.models.switch_layers import SwitchGLU

        switch_call = SwitchGLU.__call__

        def call_and_record(self, x, indices):
            routing.append(indices)
            return switch_call(self, x, indices)

        SwitchGLU.__call__ = call_and_record

    def recorded_start(self, batch, prefill_reqs, decode_reqs, scheduler_output):
        wall, t, cpu = time.time(), time.perf_counter(), time.thread_time()
        routing.clear()
        start_forward(self, batch, prefill_reqs, decode_reqs, scheduler_output)
        record = {
            "wall": wall,
            "t": t,
            "build_ms": (time.perf_counter() - t) * 1e3,
            "build_cpu_ms": (time.thread_time() - cpu) * 1e3,
            "rows": len(decode_reqs),
            "prefill": sum(len(r.token_ids) for r in prefill_reqs),
            "ctx": sum(len(state.token_ids) for _, state in decode_reqs),
        }
        if count_experts and routing and decode_reqs:
            mx.eval(routing)
            rows = len(decode_reqs)
            record["experts"] = [
                int(np.unique(np.array(i).reshape(-1, i.shape[-1])[:rows]).size)
                for i in routing
            ]
        routing.clear()
        out.write(json.dumps(record) + "\n")

    runner._start_paged_forward = recorded_start
