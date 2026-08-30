__global__ void SelectedInputTransformKernel(
    const float* input, const std::uint16_t* all_suh,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t input_stride, std::uint64_t logical_elements,
    std::uint64_t physical_elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= physical_elements) return;
  const unsigned slot = blockIdx.y;
  const std::uint32_t expert = selected_experts[slot];
  if (expert >= kTrellis35ExpertCount) return;
  input += static_cast<std::uint64_t>(slot) * input_stride;
  const std::uint16_t* suh =
      all_suh + static_cast<std::uint64_t>(expert) * physical_elements;
  output += static_cast<std::uint64_t>(slot) * physical_elements;
  const std::uint64_t block = index & ~std::uint64_t{127U};
  const unsigned column = static_cast<unsigned>(index & 127U);
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < 128U; ++row) {
    const std::uint64_t source = block + row;
    const float value = source < logical_elements ? input[source] : 0.0F;
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(value * F16(suh + source), sign, accumulator);
  }
  output[index] = accumulator * kHadamardScale;
}

__global__ void SelectedOutputTransformKernel(
    const float* input, const std::uint16_t* all_svh,
    const std::uint32_t* selected_experts, float* output,
    std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const unsigned slot = blockIdx.y;
  const std::uint32_t expert = selected_experts[slot];
  if (expert >= kTrellis35ExpertCount) return;
  input += static_cast<std::uint64_t>(slot) * elements;
  output += static_cast<std::uint64_t>(slot) * elements;
  const std::uint16_t* svh =
      all_svh + static_cast<std::uint64_t>(expert) * elements;
  const std::uint64_t block = index & ~std::uint64_t{127U};
  const unsigned column = static_cast<unsigned>(index & 127U);
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < 128U; ++row) {
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(input[block + row], sign, accumulator);
  }
  output[index] = accumulator * kHadamardScale * F16(svh + index);
}

__global__ void GatedGeluKernel(const float* gate_up, float* product) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kTrellis35ExpertIntermediate) return;
  const unsigned slot = blockIdx.y;
  gate_up += static_cast<std::uint64_t>(slot) * kTrellis35GateUpOutput;
  product +=
      static_cast<std::uint64_t>(slot) * kTrellis35ExpertIntermediate;
  const float gate = gate_up[index];
  const float up = gate_up[index + kTrellis35ExpertIntermediate];
  const float gelu =
      0.5F * gate *
      (1.0F + tanhf(kGeluScale * (gate + kGeluCubic * gate * gate * gate)));
  product[index] = gelu * up;
}

__global__ void GatedGeluBf16Kernel(const std::uint16_t* gate_up,
                                    std::uint16_t* product) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kTrellis35ExpertIntermediate) return;
  const unsigned slot = blockIdx.y;
  gate_up += static_cast<std::uint64_t>(slot) * kTrellis35GateUpOutput;
  product +=
      static_cast<std::uint64_t>(slot) * kTrellis35ExpertIntermediate;
  const float gate = Bf16(gate_up[index]);
  const float up = Bf16(gate_up[index + kTrellis35ExpertIntermediate]);
  const float gelu =
      0.5F * gate *
      (1.0F + tanhf(kGeluScale * (gate + kGeluCubic * gate * gate * gate)));
  product[index] = Bf16Bits(gelu * up);
}

__global__ void SlotOrderedReductionKernel(const float* expert_output,
                                           const float* route_weights,
                                           float* output) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kTrellis35DownOutput) return;
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned slot = 0U; slot < kTrellis35M1TopK; ++slot) {
    accumulator = fmaf(
        route_weights[slot],
        expert_output[static_cast<std::uint64_t>(slot) *
                          kTrellis35DownOutput +
                      index],
        accumulator);
  }
  output[index] = accumulator;
}

__global__ void SlotOrderedReductionT3Kernel(const float* expert_output,
                                             const float* route_weights,
                                             float* output) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kTrellis35DownOutput) return;
  const unsigned row = blockIdx.y;
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned slot = 0U; slot < kTrellis35M1TopK; ++slot) {
    const unsigned assignment = row * kTrellis35M1TopK + slot;
    accumulator = fmaf(
        route_weights[assignment],
        expert_output[static_cast<std::uint64_t>(assignment) *
                          kTrellis35DownOutput +
                      index],
        accumulator);
  }
  output[static_cast<std::uint64_t>(row) * kTrellis35DownOutput + index] =
      accumulator;
}

__global__ void OutputTransformKernel(const float* input,
                                      const std::uint16_t* svh,
                                      float* output,
                                      std::uint64_t elements) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const std::uint64_t block = index & ~std::uint64_t{127U};
  const unsigned column = static_cast<unsigned>(index & 127U);
  float accumulator = 0.0F;
#pragma unroll
  for (unsigned row = 0U; row < 128U; ++row) {
    const float sign = (__popc(row & column) & 1U) == 0U ? 1.0F : -1.0F;
    accumulator = fmaf(input[block + row], sign, accumulator);
  }
  output[index] = accumulator * kHadamardScale * F16(svh + index);
}
