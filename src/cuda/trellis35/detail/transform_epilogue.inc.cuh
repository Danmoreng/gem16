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

__global__ void SelectedInputTransformWarpKernel(
    const float* input, const std::uint16_t* all_suh,
    const std::uint32_t* selected_experts, float* output,
    unsigned assignment_count, std::uint64_t input_stride,
    unsigned assignments_per_input, std::uint64_t logical_elements,
    std::uint64_t physical_elements) {
  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const unsigned assignment = blockIdx.y * 8U + warp;
  if (assignment >= assignment_count) return;
  const std::uint32_t expert = selected_experts[assignment];
  if (expert >= kTrellis35ExpertCount) return;
  const std::uint64_t base =
      static_cast<std::uint64_t>(blockIdx.x) * 128U + lane * 4U;
  input += static_cast<std::uint64_t>(assignment / assignments_per_input) *
           input_stride;
  const std::uint16_t* suh =
      all_suh + static_cast<std::uint64_t>(expert) * physical_elements;
  output += static_cast<std::uint64_t>(assignment) * physical_elements;
  float values[4];
#pragma unroll
  for (unsigned element = 0U; element < 4U; ++element) {
    const std::uint64_t index = base + element;
    values[element] =
        index < logical_elements ? input[index] * F16(suh + index) : 0.0F;
  }
  H128Warp(values);
#pragma unroll
  for (unsigned element = 0U; element < 4U; ++element) {
    output[base + element] = values[element];
  }
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

__global__ void SelectedOutputTransformWarpKernel(
    const float* input, const std::uint16_t* all_svh,
    const std::uint32_t* selected_experts, float* output,
    unsigned assignment_count, std::uint64_t elements) {
  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  const unsigned assignment = blockIdx.y * 8U + warp;
  if (assignment >= assignment_count) return;
  const std::uint32_t expert = selected_experts[assignment];
  if (expert >= kTrellis35ExpertCount) return;
  const std::uint64_t base =
      static_cast<std::uint64_t>(blockIdx.x) * 128U + lane * 4U;
  input += static_cast<std::uint64_t>(assignment) * elements;
  output += static_cast<std::uint64_t>(assignment) * elements;
  const std::uint16_t* svh =
      all_svh + static_cast<std::uint64_t>(expert) * elements;
  float values[4];
#pragma unroll
  for (unsigned element = 0U; element < 4U; ++element) {
    values[element] = input[base + element];
  }
  H128Warp(values);
#pragma unroll
  for (unsigned element = 0U; element < 4U; ++element) {
    output[base + element] =
        values[element] * F16(svh + base + element);
  }
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

__global__ void GatedGeluDownTransformQuantizeWarpKernel(
    const float* gate_up, const std::uint16_t* all_suh,
    const std::uint32_t* selected_experts, float* scales,
    std::uint8_t* output, unsigned assignment_count) {
  const unsigned assignment = blockIdx.x;
  if (assignment >= assignment_count) return;
  const std::uint32_t expert = selected_experts[assignment];
  const bool valid = expert < kTrellis35ExpertCount;
  const float* assignment_gate_up =
      gate_up + static_cast<std::uint64_t>(assignment) *
                    kTrellis35GateUpOutput;
  const std::uint16_t* suh =
      valid ? all_suh + static_cast<std::uint64_t>(expert) *
                            kTrellis35DownInput
            : nullptr;

  __shared__ float transformed[kTrellis35DownInput];
  __shared__ float warp_maxima[8];
  const unsigned warp = threadIdx.x >> 5U;
  const unsigned lane = threadIdx.x & 31U;
  constexpr unsigned kH128Blocks = kTrellis35DownInput / 128U;
  float local_maximum = 0.0F;
  for (unsigned block = warp; block < kH128Blocks; block += 8U) {
    const std::uint64_t base =
        static_cast<std::uint64_t>(block) * 128U + lane * 4U;
    float values[4];
#pragma unroll
    for (unsigned element = 0U; element < 4U; ++element) {
      const std::uint64_t index = base + element;
      float transformed_source = 0.0F;
      if (valid && index < kTrellis35ExpertIntermediate) {
        const float gate = assignment_gate_up[index];
        const float up = assignment_gate_up[
            index + kTrellis35ExpertIntermediate];
        const float gelu =
            0.5F * gate *
            (1.0F +
             tanhf(kGeluScale *
                   (gate + kGeluCubic * gate * gate * gate)));
        const float product = gelu * up;
        transformed_source = product * F16(suh + index);
      }
      values[element] = transformed_source;
    }
    H128Warp(values);
#pragma unroll
    for (unsigned element = 0U; element < 4U; ++element) {
      transformed[base + element] = values[element];
      local_maximum = fmaxf(local_maximum, fabsf(values[element]));
    }
  }
  for (unsigned offset = 16U; offset != 0U; offset >>= 1U) {
    local_maximum = fmaxf(
        local_maximum,
        __shfl_down_sync(0xffffffffU, local_maximum, offset));
  }
  if (lane == 0U) warp_maxima[warp] = local_maximum;
  __syncthreads();
  if (warp == 0U) {
    float block_maximum = lane < 8U ? warp_maxima[lane] : 0.0F;
    for (unsigned offset = 16U; offset != 0U; offset >>= 1U) {
      block_maximum = fmaxf(
          block_maximum,
          __shfl_down_sync(0xffffffffU, block_maximum, offset));
    }
    if (lane == 0U) {
      scales[assignment] =
          block_maximum == 0.0F ? 1.0F
                                : block_maximum / kE4M3Maximum;
    }
  }
  __syncthreads();
  for (std::uint64_t index = threadIdx.x; index < kTrellis35DownInput;
       index += blockDim.x) {
    output[static_cast<std::uint64_t>(assignment) * kTrellis35DownInput +
           index] =
        __nv_fp8_e4m3(transformed[index] / scales[assignment]).__x;
  }
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
