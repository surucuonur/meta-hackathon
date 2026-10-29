---
status: accepted
---

# The baseline serves MobileMoE-S-QAT (int4), not the bfloat16 SFT model

The baseline sweep serves Meta's int4 MobileMoE-S-QAT, converted to MLX's quantized format. On the 16 GB Mac, the 2.5 GB bfloat16 SFT model left too little memory for the attention cache that 32 requests of 1,280 tokens need (about 0.8 GB), and vllm-metal refused to start because macOS swapped while it reserved that cache. The int4 weights take 0.76 GB, and int4 is also the precision Meta trained MobileMoE to run at on devices.

Meta's checkpoint stores the routed experts as [experts, in, out], grouped along out. MLX's standard quantized expert layer expects [experts, out, in] grouped along in, and a 32-value group along out cannot be re-expressed as groups along in without changing the weights. So the port keeps Meta's layout and runs experts with MLX's non-transposed quantized matmul (`transposed_experts`), which is exact. `bench/convert_qat.py` re-encodes the rest without changing any value. The served copy rounds the float16 scales to bfloat16, because MLX computes in the scale dtype and mixed dtypes promote to float32.

## Consequences

- Expert matmuls in the baseline use MLX's non-transposed quantized kernels, a different path from mlx-community 4-bit models. Its speed is part of what the profile measures.
- bfloat16 scales are slightly less accurate than Meta's reference in bfloat16 (worst error ratio 1.43 against a 1.5 limit; see results/port/).
- SFT results stay valid as a correctness check, but no bfloat16 speed baseline exists unless more memory is freed.

## Considered Options

- **Keep the SFT model and free memory:** close every other app on each run. Fragile on a 16 GB laptop, and still close to the swap limit.
- **Requantize the SFT model with `mlx_lm.convert -q`:** standard MLX layout and kernels, but not the weights Meta trained for int4.
- **Requantize the QAT experts along their input dimension:** standard kernels, but it changes the trained weights.
