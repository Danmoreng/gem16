# Glossary

**A4B**
A model label indicating total and active parameter scale; for Gemma 4 26B A4B, only a subset of expert parameters is active per token, but all weights must remain resident for fast local inference.

**BF16**
16-bit brain floating point. Used for sensitive small tensors, router/norms and high-precision references.

**CUDA Graph**
Captured fixed-address GPU execution graph replayed with low launch overhead.

**E2M1**
Signed 4-bit floating-point element format used by NVFP4 values.

**E4M3 / E4M3FN**
8-bit floating-point format used for FP8 weights and NVFP4 block scales. Exact finite/NaN behavior must be codec-specific.

**FP8**
8-bit floating-point. The plan uses FP8 attention weights and dynamic per-token activation quantization.

**GGUF**
File format used by llama.cpp and the official Google Q4_0 release. It is a reference input, not necessarily the production gem16 artifact format.

**ITL**
Inter-token latency.

**KV cache**
Cached attention key/value states. Gemma uses local sliding rings and global full extents.

**MMA**
Matrix multiply-accumulate instruction, including native Blackwell block-scaled narrow-precision Tensor Core operations.

**MoE**
Mixture of Experts. A router selects eight of 128 routed experts per token. The 26B model also has an always-active shared dense MLP.

**MTP**
Multi-token prediction/speculative assistant path. Not in the first 26B release.

**NVFP4**
NVIDIA FP4 block-scaled format using E2M1 values and UE4M3/E4M3 scales with vector size 16 for the target native Blackwell path.

**PTQ**
Post-training quantization.

**Q4_0**
llama.cpp-style 4-bit block format with one FP16 scale and 16 packed bytes for 32 weights.

**QAT**
Quantization-aware training. Google's released QAT weights were trained toward Q4_0 behavior; converting them to NVFP4 remains an empirical experiment.

**Row8/K64**
gem16's current runtime order for direct SM120 NVFP4 access: row tiles of 8 and K blocks of 64.

**Safetensors**
Non-executable tensor container used for source and proposed derived artifacts.

**SM120/SM120a**
Blackwell GeForce compute target used by the first optimized gem16 backend.

**SQNR**
Signal-to-quantization-noise ratio.

**TTFT**
Time to first token.

**W4A4**
4-bit weights and 4-bit activations, used by native NVFP4 expert execution.

**W4A16**
4-bit weights with BF16/FP16-like activations, a likely Q4_0 output-head strategy.
