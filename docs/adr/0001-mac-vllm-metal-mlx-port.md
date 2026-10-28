---
status: accepted
---

# Study vLLM on the Mac through vllm-metal, with MobileMoE ported to mlx_lm

We measure vLLM serving MobileMoE-S on an M1 Pro through the vllm-metal plugin, not standard vLLM on rented NVIDIA GPUs. MobileMoE is built for on-device use, the Mac is free and always available, and the question is about unified-memory bandwidth. As a result, the code that decides speed is mlx_lm (the model's layers) and vllm-metal (attention), not vLLM's CUDA kernels, so profiling targets those. MobileMoE, which no engine supported, gets ported into mlx_lm as `mlx_lm/models/mobilemoe.py`.

## Considered Options

- **Standard vLLM on NVIDIA GPUs through Modal credits:** vLLM's own kernels would be the hot path, but it means paid, remote work and drops the on-device angle.
- **vllm-metal's PyTorch MPS path with Meta's custom model code:** no port if it worked, but that's uncertain, and it isn't the path vllm-metal optimizes.
