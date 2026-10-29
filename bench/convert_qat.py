"""Convert Meta's int4 MobileMoE-S-QAT checkpoint to MLX's quantized format.

    HF_HUB_OFFLINE=1 .venv/bin/python bench/convert_qat.py                   # serving copy
    HF_HUB_OFFLINE=1 .venv/bin/python bench/convert_qat.py --dtype float32   # exact copy

Meta stores signed 4-bit values (-8..7), two per byte with the low nibble
first, and one float16 scale per group of 32 values. MLX stores unsigned
4-bit values, eight per uint32 word with the lowest bits first, plus a scale
and a bias per group. Flipping the top bit of each nibble maps -8..7 to
0..15, so scale * q = scale * (q + 8) - 8 * scale: the bias is -8 * scale and
no value changes. The bytes are then read as uint32 words in place.

Attention, the shared expert and the embeddings are grouped along their input
dimension, as MLX's quantized layers expect. The routed experts are stored as
[E, in, out] and grouped along out, so the port runs them as x @ W
(`transposed_experts` in mlx_lm/models/mobilemoe.py).

--dtype sets the dtype of the scales and biases, and MLX computes in that
dtype. The serving copy uses bfloat16, which rounds each float16 scale once.
The float32 copy keeps the scales exact, for the float32 correctness gate in
check_port.py. Output: models/<name>-mlx-<dtype>/ (git-ignored).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
META_FORMAT = "mobilemoe-int4-g32"
SKIP = {".gitattributes", "README.md", "config.json"}


def model_dir(model: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model, allow_patterns=["*.json", "*.py", "*.safetensors", "LICENSE"]
        )
    )


def to_mlx(packed: mx.array, scale: mx.array, dtype) -> dict:
    """Meta int4 bytes and float16 scales -> MLX weight, scales and biases."""
    scales = scale.astype(dtype)
    return {
        "weight": mx.bitwise_xor(packed, mx.array(0x88, mx.uint8)).view(mx.uint32),
        "scales": scales,
        "biases": -8 * scales,
    }


def convert(weights: dict, dtype) -> dict:
    out = {}
    for name, value in weights.items():
        prefix, _, leaf = name.rpartition(".")
        if leaf == "gate_up_qweight":
            scale = weights[f"{prefix}.gate_up_scale"]
            q, s = value.shape[-1] // 2, scale.shape[-1] // 2
            # The gate is the first half of the fused output.
            halves = {
                "gate_proj": (value[..., :q], scale[..., :s]),
                "up_proj": (value[..., q:], scale[..., s:]),
            }
            for proj, (packed, part) in halves.items():
                for key, tensor in to_mlx(packed, part, dtype).items():
                    out[f"{prefix}.{proj}.{key}"] = tensor
        elif leaf == "down_qweight":
            scale = weights[f"{prefix}.down_scale"]
            for key, tensor in to_mlx(value, scale, dtype).items():
                out[f"{prefix}.down_proj.{key}"] = tensor
        elif leaf == "qweight":
            scale = weights[f"{prefix}.weight_scale"]
            for key, tensor in to_mlx(value, scale, dtype).items():
                out[f"{prefix}.{key}"] = tensor
        elif leaf not in ("gate_up_scale", "down_scale", "weight_scale"):
            out[name] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="facebook/MobileMoE-S-QAT")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    src = model_dir(args.model)
    config = json.loads((src / "config.json").read_text())
    meta_quant = config.pop("quantization")
    expected = {"format": META_FORMAT, "packed": True, "qmin": -8, "qmax": 7}
    if any(meta_quant.get(k) != v for k, v in expected.items()):
        sys.exit(f"Unsupported quantization block: {meta_quant}")
    if meta_quant["embedding_group_size"] != meta_quant["group_size"]:
        sys.exit("Different embedding and linear group sizes are not supported.")

    dst = ROOT / "models" / f"{args.model.split('/')[-1]}-mlx-{args.dtype}"
    if dst.exists():
        sys.exit(f"{dst} exists; delete it to convert again.")
    dst.mkdir(parents=True)

    weights = {}
    for path in sorted(src.glob("*.safetensors")):
        weights.update(mx.load(str(path)))
    weights = convert(weights, getattr(mx, args.dtype))
    mx.save_safetensors(
        str(dst / "model.safetensors"), weights, metadata={"format": "mlx"}
    )

    config["quantization"] = {
        "group_size": meta_quant["group_size"],
        "bits": 4,
        "mode": "affine",
    }
    config["transposed_experts"] = True
    config["converted_from"] = {
        "model": args.model,
        "revision": src.name,
        "scale_dtype": args.dtype,
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for path in src.iterdir():
        if path.name not in SKIP and path.suffix not in (".safetensors", ".png"):
            shutil.copy(path, dst / path.name)

    # Fail now, not at serving time, if a tensor name or shape is wrong.
    from mlx_lm.utils import load_model

    load_model(dst)
    print(f"saved {dst} ({len(weights)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
