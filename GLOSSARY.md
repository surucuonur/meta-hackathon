# MobileMoE Bandwidth Study

How efficiently an inference engine turns an Apple Silicon GPU's memory bandwidth and compute into generated tokens when serving a small mixture-of-experts model.

## Hardware limits

**Peak bandwidth**:
The most bytes per second the GPU can read from unified memory, as rated by the manufacturer (about 200 GB/s on this project's M1 Pro).
_Avoid_: max bandwidth, memory speed

**Peak compute**:
The most floating-point operations per second the GPU can perform (about 5.2 TFLOPS on the 16-core M1 Pro GPU).
_Avoid_: max compute, TFLOPS (as the name of the limit)

**Achieved bandwidth**:
The bytes a decode step must read from memory (the weights it touches plus its share of cached attention state), divided by that step's duration.
_Avoid_: bandwidth usage, utilization (unqualified)

**Arithmetic intensity**:
Floating-point operations performed per byte read from memory during a decode step.
_Avoid_: compute density, FLOPs/byte ratio

**Ridge point**:
The arithmetic intensity at which peak bandwidth and peak compute meet (about 26 operations per byte here); below it a step is limited by bandwidth, above it by compute.

**Roofline**:
The chart of achievable throughput against arithmetic intensity, capped by peak bandwidth on the left and peak compute on the right, with measured decode steps plotted beneath it.

## Bottlenecks and fixes

**Bandwidth-bound**:
A decode step whose achieved bandwidth is close to peak bandwidth.

**Compute-bound**:
A decode step whose operation rate is close to peak compute.

**Overhead-bound**:
A decode step far below both limits, because its time goes to work other than reading weights or doing arithmetic, such as launching GPU jobs, routing tokens to experts or scheduling.
_Avoid_: latency-bound, CPU-bound (unless the CPU specifically is the cause)

**Bottleneck**:
The single cause, identified from measurements, that most limits throughput for a given workload.
_Avoid_: issue, slowness

**Baseline**:
The unmodified engine serving MobileMoE-S-QAT (int4), against which every fix is measured.

**Fix**:
One change to the engine, the model code or the configuration, measured against the baseline under the same workload.
_Avoid_: optimization (for a change that hasn't been measured), tweak

## Inference

**Prefill**:
Processing a request's whole prompt in one pass before any token is generated.

**Decode step**:
One engine iteration that produces one new token for each request it includes.

**Decode rows**:
The number of requests the engine actually advanced in one decode step; the batch size that matters for the roofline.
_Avoid_: batch size (unqualified), concurrency

**Client concurrency**:
The number of requests the benchmark client keeps in flight; it caps decode rows but does not equal them.
_Avoid_: batch size

## Mixture of experts

**Expert**:
One of the interchangeable feed-forward blocks in a layer; MobileMoE-S has 60 routed experts per layer.

**Active experts**:
The experts the router picks for one token in one layer (4 for MobileMoE-S).
_Avoid_: selected experts, top-k experts

**Distinct experts**:
The number of different experts used by all decode rows together in one layer during one decode step; it decides how much expert weight data the step must read.
_Avoid_: unique experts, expert coverage

## Models

**MobileMoE-S**:
Meta's smallest on-device mixture-of-experts model (about 1.3B parameters in total, about 272M used per token), studied here in its SFT (full-precision) and QAT (int4) variants.

**Muse Spark**:
Meta's hosted model, used in this project only to explain profiles and source code, never as the model under study.
