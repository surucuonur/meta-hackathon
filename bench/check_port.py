"""Check the MLX port of MobileMoE against Meta's PyTorch reference.

Meta's model code needs transformers 4.57 and mlx-lm needs 5.7 or newer, so
the check runs in two steps, one per environment:

    .venv-ref/bin/python bench/check_port.py ref --dtype float32
    .venv/bin/python     bench/check_port.py mlx --dtype float32

`ref` runs Meta's code on fixed prompts and saves the token ids, its greedy
output and its logits. `mlx` runs the port on the same token ids, compares,
and writes a summary to results/port/. Run float32 first: the bfloat16 check
uses the port's float32 logits as the exact answer.

Pass criteria:
  float32   the correctness gate. Every prompt: same greedy tokens, same
            top-1 token at every position, max |logit difference| <= 1e-2.
  bfloat16  a sanity gate for the serving dtype. Every prompt: the port's
            mean |logit error| against the exact answer is at most 1.5x the
            reference's own bfloat16 error. Greedy tokens are reported but
            not gated: bfloat16 logits can tie, and either tied token is
            correct. This gate catches bfloat16-only breakage (overflow,
            garbage output), not small precision changes: computing the
            router in bfloat16 instead of float32 stays within 1.2x.
  both      the serving tokenizer produces the reference's token ids.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "port"
EOS_IDS = {128001, 128007, 128008}
FP32_MAX_ABS_DIFF = 1e-2
BF16_MAX_ERROR_RATIO = 1.5
PROMPTS = [
    {"chat": "Explain why the sky is blue in a short paragraph."},
    {"chat": "Write a Python function that checks whether a number is prime."},
    {"chat": "List three differences between a CPU and a GPU."},
    {"text": "The history of the printing press begins"},
]


def model_dir(model: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(model, allow_patterns=["*.json", "*.py", "*.safetensors"])
    )


def encode(tokenizer, prompt: dict) -> list[int]:
    if "chat" in prompt:
        messages = [{"role": "user", "content": prompt["chat"]}]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return list(tokenizer.encode(text, add_special_tokens=False))
    return list(tokenizer.encode(prompt["text"]))


def ref_file(model: str, dtype: str) -> Path:
    return OUT_DIR / f"{model.split('/')[-1]}-{dtype}-reference.npz"


def summary_file(model: str, dtype: str) -> Path:
    return OUT_DIR / f"{model.split('/')[-1]}-{dtype}.json"


def git_commit(package_file: str) -> str | None:
    """Commit of the checkout a package runs from, with -dirty for local edits."""
    repo = Path(package_file).resolve().parent.parent

    def git(*cmd: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *cmd], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = git("rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        return None
    return sha + ("-dirty" if git("status", "--porcelain") else "")


def run_ref(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = model_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="eager",
    ).eval()

    def greedy(ids: list[int]) -> list[int]:
        out = model(torch.tensor([ids]), use_cache=True)
        generated = []
        for _ in range(args.max_tokens):
            token = int(out.logits[0, -1].argmax())
            generated.append(token)
            if token in EOS_IDS:
                break
            out = model(
                torch.tensor([[token]]),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
        return generated

    arrays = {"revision": np.array(path.name)}
    with torch.inference_mode():
        for i, prompt in enumerate(PROMPTS):
            ids = encode(tokenizer, prompt)
            generated = greedy(ids)
            sequence = torch.tensor([ids + generated])
            logits = model(sequence, use_cache=False).logits[0].float().numpy()
            arrays[f"{i}.prompt"] = np.array(ids)
            arrays[f"{i}.generated"] = np.array(generated)
            arrays[f"{i}.logits"] = logits
            print(f"prompt {i}: {len(ids)} prompt + {len(generated)} generated tokens")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(ref_file(args.model, args.dtype), **arrays)
    print(f"saved {ref_file(args.model, args.dtype)}")


def compare(ref_logits, logits, ref_generated, generated, prompt_len) -> dict:
    diff = np.abs(ref_logits - logits)
    k = 5
    ref_top = np.argpartition(-ref_logits, k, axis=-1)[:, :k]
    top = np.argpartition(-logits, k, axis=-1)[:, :k]
    overlap = [len(set(a) & set(b)) / k for a, b in zip(ref_top, top)]

    divergence = next(
        (j for j, (a, b) in enumerate(zip(ref_generated, generated)) if a != b),
        None,
    )
    if divergence is None and len(ref_generated) != len(generated):
        divergence = min(len(ref_generated), len(generated))
    margin = None
    if divergence is not None:
        # Logits at position t predict token t + 1.
        row = np.sort(ref_logits[prompt_len - 1 + divergence])
        margin = float(row[-1] - row[-2])

    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "top1_agreement": float(
            (ref_logits.argmax(-1) == logits.argmax(-1)).mean()
        ),
        "top5_overlap": float(np.mean(overlap)),
        "greedy_match": divergence is None,
        "first_divergence": divergence,
        "ref_top2_margin_at_divergence": margin,
    }


def run_mlx(args) -> int:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    if args.dtype == "bfloat16":
        fp32 = summary_file(args.model, "float32")
        if not fp32.exists() or json.loads(fp32.read_text())["verdict"] != "PASS":
            sys.exit(f"Run and pass the float32 check first ({fp32} missing or FAIL).")

    ref = np.load(ref_file(args.model, args.dtype))
    path = model_dir(args.model)
    model, tokenizer = load(
        str(path), tokenizer_config={"trust_remote_code": True}
    )
    if args.dtype == "float32":
        model.set_dtype(mx.float32)

    def logits_for(sequence: list[int]) -> np.ndarray:
        return np.array(model(mx.array([sequence]))[0].astype(mx.float32))

    def greedy(ids: list[int]) -> list[int]:
        cache = make_prompt_cache(model)
        logits = model(mx.array([ids]), cache=cache)
        generated = []
        for _ in range(args.max_tokens):
            token = int(logits[0, -1].argmax())
            generated.append(token)
            if token in EOS_IDS:
                break
            logits = model(mx.array([[token]]), cache=cache)
        return generated

    results, sequences = [], []
    for i, prompt in enumerate(PROMPTS):
        ids = ref[f"{i}.prompt"].tolist()
        ref_generated = ref[f"{i}.generated"].tolist()
        sequences.append(ids + ref_generated)
        result = compare(
            ref[f"{i}.logits"],
            logits_for(sequences[-1]),
            ref_generated,
            greedy(ids),
            len(ids),
        )
        result["tokenizer_match"] = encode(tokenizer, prompt) == ids
        result["prompt"] = prompt
        results.append(result)

    if args.dtype == "bfloat16":
        port = [logits_for(s) for s in sequences]
        model.set_dtype(mx.float32)
        for i, (r, s) in enumerate(zip(results, sequences)):
            exact = logits_for(s)
            r["ref_error"] = float(np.abs(ref[f"{i}.logits"] - exact).mean())
            r["port_error"] = float(np.abs(port[i] - exact).mean())
            r["error_ratio"] = r["port_error"] / r["ref_error"]

    def passed(r: dict) -> bool:
        if not r["tokenizer_match"]:
            return False
        if args.dtype == "bfloat16":
            return r["error_ratio"] <= BF16_MAX_ERROR_RATIO
        return (
            r["greedy_match"]
            and r["top1_agreement"] == 1.0
            and r["max_abs_diff"] <= FP32_MAX_ABS_DIFF
        )

    verdict = "PASS" if all(passed(r) for r in results) else "FAIL"
    ratio_header = f" {'err ratio':>9}" if args.dtype == "bfloat16" else ""
    print(f"{'#':>2} {'max|d|':>8} {'mean|d|':>9} {'top1':>6} {'top5':>6}"
          f"{ratio_header} {'tok':>4}  greedy")
    for i, r in enumerate(results):
        greedy_note = "same" if r["greedy_match"] else (
            f"differs at token {r['first_divergence']} "
            f"(reference top-2 margin {r['ref_top2_margin_at_divergence']:.3f})"
        )
        ratio = f" {r['error_ratio']:9.2f}" if "error_ratio" in r else ""
        print(f"{i:>2} {r['max_abs_diff']:8.4f} {r['mean_abs_diff']:9.5f} "
              f"{r['top1_agreement']:6.3f} {r['top5_overlap']:6.3f}{ratio} "
              f"{'ok' if r['tokenizer_match'] else 'DIFF':>4}  {greedy_note}")
    print(f"{args.dtype}: {verdict}")

    summary = {
        "model": args.model,
        "revision": str(ref["revision"]),
        "dtype": args.dtype,
        "verdict": verdict,
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
        "mlx_lm_commit": git_commit(mlx_lm.__file__),
        "results": results,
    }
    out = summary_file(args.model, args.dtype)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved {out}")
    return 0 if verdict == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["ref", "mlx"])
    parser.add_argument("--model", default="facebook/MobileMoE-S-SFT")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    if args.step == "ref":
        run_ref(args)
        return 0
    return run_mlx(args)


if __name__ == "__main__":
    sys.exit(main())
