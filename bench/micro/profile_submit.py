"""Investigation: where does vllm-metal's per-step CPU time go at one request?

In-process engine (no multiprocessing) so cProfile sees the model runner.
"""

import cProfile
import os
import pstats
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

from vllm import LLM, SamplingParams  # noqa: E402

out_dir = sys.argv[1]
llm = LLM(
    model="models/MobileMoE-S-QAT-mlx-bfloat16",
    trust_remote_code=True,
    max_model_len=1024,
    max_num_seqs=4,
    kv_cache_memory_bytes=256 << 20,
    enable_prefix_caching=False,
    generation_config="vllm",
)
sp = SamplingParams(max_tokens=256, temperature=0, ignore_eos=True)
prompt = {"prompt_token_ids": list(range(1000, 1128))}
llm.generate([prompt], sp, use_tqdm=False)  # warm-up

t0 = time.perf_counter()
llm.generate([prompt], sp, use_tqdm=False)
plain = time.perf_counter() - t0
print(
    f"unprofiled request: {plain * 1e3:.0f} ms for 256 tokens = {plain / 256 * 1e3:.2f} ms/token"
)

prof = cProfile.Profile()
prof.enable()
llm.generate([prompt], sp, use_tqdm=False)
prof.disable()
prof.dump_stats(f"{out_dir}/step.prof")
stats = pstats.Stats(prof)
stats.sort_stats("cumulative").print_stats(60)
stats.sort_stats("tottime").print_stats(40)
