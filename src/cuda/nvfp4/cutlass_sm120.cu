#include "cuda/nvfp4/cutlass_sm120.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#if defined(_WIN32)
#include "cuda/cutlass_windows_launch.cuh"
#endif

namespace gem16::internal {

struct GatedGeluScaledProductArguments {
  float up_scale = 1.0F;
};

template <class T>
struct GatedGeluScaledProduct {
  using Arguments = GatedGeluScaledProductArguments;

  CUTLASS_DEVICE T operator()(T gate, T up_accumulator,
                              const Arguments& arguments) const {
    if constexpr (std::is_same_v<T, float>) {
      return Evaluate(gate, up_accumulator, arguments);
    } else {
      T output;
#pragma unroll
      for (int index = 0; index < T::kElements; ++index) {
        output[index] =
            Evaluate(gate[index], up_accumulator[index], arguments);
      }
      return output;
    }
  }

 private:
  CUTLASS_DEVICE static float Evaluate(float gate, float up_accumulator,
                                       const Arguments& arguments) {
    constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
    constexpr float kGeluCubic = 0.044715F;
    const float rounded_up = static_cast<float>(
        __float2bfloat16_rn(up_accumulator * arguments.up_scale));
    const float inner = kSqrtTwoOverPi *
                        (gate + kGeluCubic * gate * gate * gate);
    const float gelu = static_cast<float>(__float2bfloat16_rn(
        0.5F * gate * (1.0F + tanhf(inner))));
    const float product =
        static_cast<float>(__float2bfloat16_rn(gelu * rounded_up));
    return product;
  }
};

struct GatedGeluNvfp4Fusion
    : cutlass::epilogue::fusion::FusionOperation {
  using ElementOutput = cutlass::float_e2m1_t;
  using ElementCompute = float;
  using ElementSource = cutlass::bfloat16_t;
  static constexpr bool IsSourceSupported = true;
  using ElementBlockScaleFactor = cutlass::float_ue4m3_t;
  static constexpr int SFVecSize = 16;
  static constexpr bool IsBlockScaleSupported = true;
  using GmemLayoutTagScalefactor = cutlass::layout::RowMajor;
};

}  // namespace gem16::internal

namespace cutlass::epilogue::fusion {

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC,
          bool DelayTmaStore, class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm120TmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                     ReuseSmemC, DelayTmaStore>,
    gem16::internal::GatedGeluNvfp4Fusion, CtaTileShapeMNK, EpilogueTile>
    : Sm90EVT<
          Sm120BlockScaleFactorRowStore<
              16, EpilogueTile, CtaTileShapeMNK, FragmentSize,
              cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t,
              cutlass::FloatRoundStyle::round_to_nearest>,
          Sm90EVT<
              Sm90Compute<gem16::internal::GatedGeluScaledProduct, float,
                          float,
                          cutlass::FloatRoundStyle::round_to_nearest>,
              Sm90SrcFetch<cutlass::bfloat16_t>, Sm90AccFetch>> {
  using Impl = Sm90EVT<
      Sm120BlockScaleFactorRowStore<
          16, EpilogueTile, CtaTileShapeMNK, FragmentSize,
          cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t,
          cutlass::FloatRoundStyle::round_to_nearest>,
      Sm90EVT<
          Sm90Compute<gem16::internal::GatedGeluScaledProduct, float, float,
                      cutlass::FloatRoundStyle::round_to_nearest>,
          Sm90SrcFetch<cutlass::bfloat16_t>, Sm90AccFetch>>;
  using Operation = gem16::internal::GatedGeluNvfp4Fusion;

  struct Arguments {
    float up_scale = 1.0F;
    cutlass::float_ue4m3_t* block_scale_factor_ptr = nullptr;
    const float* norm_constant_ptr = nullptr;

    operator typename Impl::Arguments() const {
      return {
          {{}, {}, {up_scale}},
          {block_scale_factor_ptr, norm_constant_ptr,
           {cute::_0{}, cute::_0{}, 0}}};
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion

namespace gem16::internal {
namespace {

using namespace cute;

constexpr std::uint64_t kRowsPerDecodeTile = 8U;
constexpr std::uint64_t kElementsPerKBlock = 64U;
constexpr std::uint64_t kPackedBytesPerKBlock = 32U;
constexpr std::uint64_t kScalesPerKBlock = 4U;
constexpr std::uint64_t kScaleRowsPerTile = 128U;
constexpr std::uint64_t kScaleColumnsPerTile = 4U;
constexpr unsigned kThreads = 256U;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status Internal(std::string message) {
  return Status(StatusCode::kInternal, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Internal(std::string(operation) + ": " + cudaGetErrorName(error) +
                  ": " + cudaGetErrorString(error));
}

bool PositiveFinite(float value) {
  return std::isfinite(value) && value > 0.0F;
}

__device__ __forceinline__ std::uint64_t InterleavedScaleOffset(
    std::uint64_t row, std::uint64_t column,
    std::uint64_t padded_columns) {
  const std::uint64_t scale_k_tiles =
      padded_columns / kScaleColumnsPerTile;
  const std::uint64_t tile_offset =
      ((row / kScaleRowsPerTile) * scale_k_tiles +
       column / kScaleColumnsPerTile) *
      (kScaleRowsPerTile * kScaleColumnsPerTile);
  return tile_offset + (row % 32U) * 16U +
         ((row % kScaleRowsPerTile) / 32U) * 4U +
         column % kScaleColumnsPerTile;
}

__global__ void InterleaveActivationScalesKernel(
    const std::uint8_t* compact, std::uint8_t* interleaved,
    std::uint64_t rows, std::uint64_t columns,
    std::uint64_t padded_rows, std::uint64_t padded_columns) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t column_groups =
      padded_columns / kScaleColumnsPerTile;
  const std::uint64_t groups = padded_rows * column_groups;
  if (index >= groups) return;
  const std::uint64_t row = index / column_groups;
  const std::uint64_t column = (index % column_groups) * kScaleColumnsPerTile;
  std::uint32_t value = 0U;
  if (row < rows && column < columns) {
    value = *reinterpret_cast<const std::uint32_t*>(
        compact + row * columns + column);
  }
  *reinterpret_cast<std::uint32_t*>(
      interleaved +
      InterleavedScaleOffset(row, column, padded_columns)) = value;
}

__global__ void PrepareWeightKernel(
    const std::uint8_t* tiled_weight,
    const std::uint8_t* tiled_scales,
    std::uint8_t* row_major_weight,
    std::uint8_t* interleaved_scales,
    std::uint64_t rows,
    std::uint64_t contracting_elements) {
  const std::uint64_t packed_row_bytes = contracting_elements / 2U;
  const std::uint64_t scale_row_bytes = contracting_elements / 16U;
  constexpr std::uint64_t kCopyBytes = sizeof(uint4);
  const std::uint64_t packed_copies =
      rows * packed_row_bytes / kCopyBytes;
  const std::uint64_t scale_groups =
      rows * scale_row_bytes / kScalesPerKBlock;
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;

  if (index < packed_copies) {
    const std::uint64_t copies_per_row = packed_row_bytes / kCopyBytes;
    const std::uint64_t row = index / copies_per_row;
    const std::uint64_t copy_in_row = index % copies_per_row;
    const std::uint64_t k_block =
        copy_in_row / (kPackedBytesPerKBlock / kCopyBytes);
    const std::uint64_t byte_in_block =
        (copy_in_row % (kPackedBytesPerKBlock / kCopyBytes)) * kCopyBytes;
    const std::uint64_t first_row =
        (row / kRowsPerDecodeTile) * kRowsPerDecodeTile;
    const std::uint64_t rows_in_tile =
        min(kRowsPerDecodeTile, rows - first_row);
    const std::uint64_t source =
        first_row * k_blocks * kPackedBytesPerKBlock +
        (k_block * rows_in_tile + row % kRowsPerDecodeTile) *
            kPackedBytesPerKBlock +
        byte_in_block;
    reinterpret_cast<uint4*>(row_major_weight)[index] =
        *reinterpret_cast<const uint4*>(tiled_weight + source);
  }

  if (index < scale_groups) {
    const std::uint64_t row = index / k_blocks;
    const std::uint64_t k_block = index % k_blocks;
    const std::uint64_t column = k_block * kScalesPerKBlock;
    const std::uint64_t first_row =
        (row / kRowsPerDecodeTile) * kRowsPerDecodeTile;
    const std::uint64_t rows_in_tile =
        min(kRowsPerDecodeTile, rows - first_row);
    const std::uint64_t source =
        first_row * k_blocks * kScalesPerKBlock +
        (k_block * rows_in_tile + row % kRowsPerDecodeTile) *
            kScalesPerKBlock;
    *reinterpret_cast<std::uint32_t*>(
        interleaved_scales +
        InterleavedScaleOffset(row, column, scale_row_bytes)) =
        *reinterpret_cast<const std::uint32_t*>(tiled_scales + source);
  }
}

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator, void,
        cutlass::layout::RowMajor, 1, ElementD,
        cutlass::layout::RowMajor, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, cutlass::layout::RowMajor, 32,
        ElementB, cutlass::layout::ColumnMajor, 32, ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
    cutlass::gemm::StaticPersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using GatedElementD = cutlass::float_e2m1_t;
using GatedCollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator, cutlass::bfloat16_t,
        cutlass::layout::RowMajor, 8, GatedElementD,
        cutlass::layout::RowMajor, 32,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        GatedGeluNvfp4Fusion>::CollectiveOp;

using GatedCollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, cutlass::layout::RowMajor, 32,
        ElementB, cutlass::layout::ColumnMajor, 32, ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(
                sizeof(typename GatedCollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GatedGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, GatedCollectiveMainloop,
    GatedCollectiveEpilogue, cutlass::gemm::StaticPersistentScheduler>;
using GatedGemm =
    cutlass::gemm::device::GemmUniversalAdapter<GatedGemmKernel>;

Status LaunchGemm(
    const std::uint8_t* activation,
    const std::uint8_t* activation_scales,
    const std::uint8_t* weight,
    const std::uint8_t* weight_scales,
    std::uint16_t* output,
    int m, int n, int k, float alpha,
    void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
  using ScaleConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  const auto problem_shape = cute::make_shape(m, n, k, 1);
  const auto stride_a = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideA{}, {m, k, 1});
  const auto stride_b = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideB{}, {n, k, 1});
  const auto stride_d = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideD{}, {m, n, 1});
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {reinterpret_cast<const cutlass::float_e2m1_t*>(activation), stride_a,
       reinterpret_cast<const cutlass::float_e2m1_t*>(weight), stride_b,
       reinterpret_cast<const cutlass::float_ue4m3_t*>(activation_scales),
       ScaleConfig::tile_atom_to_shape_SFA(problem_shape),
       reinterpret_cast<const cutlass::float_ue4m3_t*>(weight_scales),
       ScaleConfig::tile_atom_to_shape_SFB(problem_shape)},
      {{alpha, 0.0F}, nullptr, stride_d,
       reinterpret_cast<cutlass::bfloat16_t*>(output), stride_d}};

  if constexpr (!std::is_const_v<
                    decltype(arguments.scheduler.max_swizzle_size)>) {
    arguments.scheduler.max_swizzle_size = 1;
  }
  const std::size_t required = Gemm::get_workspace_size(arguments);
#if defined(_WIN32)
  constexpr std::size_t kWindowsParamsPadding =
      alignof(typename Gemm::Params) - 1U;
  if (required > workspace_bytes ||
      kWindowsParamsPadding > workspace_bytes - required ||
      sizeof(typename Gemm::Params) >
          workspace_bytes - required - kWindowsParamsPadding) {
#else
  if (required > workspace_bytes) {
#endif
    return Invalid("CUTLASS NVFP4 workspace is too small");
  }
#if defined(_WIN32)
  cutlass::Status status = cutlass_windows::InitializeAndRun<Gemm>(
      arguments, workspace, workspace_bytes, stream);
  if (status != cutlass::Status::kSuccess) {
    return Internal(std::string("CUTLASS NVFP4 Windows launch failed: ") +
                    cutlass::cutlassGetStatusString(status));
  }
#else
  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return Invalid(std::string("CUTLASS cannot implement NVFP4 projection: ") +
                   cutlass::cutlassGetStatusString(status));
  }
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return Internal(std::string("CUTLASS NVFP4 initialization failed: ") +
                    cutlass::cutlassGetStatusString(status));
  }
  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return Internal(std::string("CUTLASS NVFP4 launch failed: ") +
                    cutlass::cutlassGetStatusString(status));
  }
#endif
  return Status::Ok();
}

Status LaunchGatedGemm(
    const std::uint8_t* activation,
    const std::uint8_t* activation_scales,
    const std::uint8_t* weight,
    const std::uint8_t* weight_scales,
    const std::uint16_t* gate,
    std::uint8_t* product,
    std::uint8_t* product_scales,
    int m, int n, int k, float up_scale,
    const float* product_global_divisor,
    void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
  using ScaleConfig = typename GatedGemm::GemmKernel::CollectiveMainloop::
      Sm1xxBlkScaledConfig;
  const auto problem_shape = cute::make_shape(m, n, k, 1);
  const auto stride_a = cutlass::make_cute_packed_stride(
      typename GatedGemm::GemmKernel::StrideA{}, {m, k, 1});
  const auto stride_b = cutlass::make_cute_packed_stride(
      typename GatedGemm::GemmKernel::StrideB{}, {n, k, 1});
  const auto stride_c = cutlass::make_cute_packed_stride(
      typename GatedGemm::GemmKernel::StrideC{}, {m, n, 1});
  const auto stride_d = cutlass::make_cute_packed_stride(
      typename GatedGemm::GemmKernel::StrideD{}, {m, n, 1});
  typename GatedGemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {reinterpret_cast<const cutlass::float_e2m1_t*>(activation), stride_a,
       reinterpret_cast<const cutlass::float_e2m1_t*>(weight), stride_b,
       reinterpret_cast<const cutlass::float_ue4m3_t*>(activation_scales),
       ScaleConfig::tile_atom_to_shape_SFA(problem_shape),
       reinterpret_cast<const cutlass::float_ue4m3_t*>(weight_scales),
       ScaleConfig::tile_atom_to_shape_SFB(problem_shape)},
      {{up_scale,
        reinterpret_cast<cutlass::float_ue4m3_t*>(product_scales),
        product_global_divisor},
       reinterpret_cast<const cutlass::bfloat16_t*>(gate), stride_c,
       reinterpret_cast<GatedElementD*>(product), stride_d}};

  if constexpr (!std::is_const_v<
                    decltype(arguments.scheduler.max_swizzle_size)>) {
    arguments.scheduler.max_swizzle_size = 1;
  }
  const std::size_t required = GatedGemm::get_workspace_size(arguments);
#if defined(_WIN32)
  constexpr std::size_t kWindowsParamsPadding =
      alignof(typename GatedGemm::Params) - 1U;
  if (required > workspace_bytes ||
      kWindowsParamsPadding > workspace_bytes - required ||
      sizeof(typename GatedGemm::Params) >
          workspace_bytes - required - kWindowsParamsPadding) {
#else
  if (required > workspace_bytes) {
#endif
    return Invalid("CUTLASS gated NVFP4 workspace is too small");
  }
#if defined(_WIN32)
  const cutlass::Status status =
      cutlass_windows::InitializeAndRun<GatedGemm>(
          arguments, workspace, workspace_bytes, stream);
#else
  GatedGemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status == cutlass::Status::kSuccess) {
    status = gemm.initialize(arguments, workspace, stream);
  }
  if (status == cutlass::Status::kSuccess) status = gemm.run(stream);
#endif
  if (status != cutlass::Status::kSuccess) {
    return Internal(std::string("CUTLASS gated NVFP4 launch failed: ") +
                    cutlass::cutlassGetStatusString(status));
  }
  return Status::Ok();
}

}  // namespace

Status LaunchNvfp4CutlassInterleaveActivationScales(
    const std::uint8_t* compact_scales_e4m3fn,
    std::uint8_t* interleaved_scales_e4m3fn,
    std::uint64_t tokens,
    std::uint64_t contracting_elements,
    cudaStream_t stream) {
  if (compact_scales_e4m3fn == nullptr ||
      interleaved_scales_e4m3fn == nullptr || tokens == 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("invalid CUTLASS activation-scale interleave arguments");
  }
  const std::uint64_t columns = contracting_elements / 16U;
  const std::uint64_t padded_rows =
      ((tokens + kScaleRowsPerTile - 1U) / kScaleRowsPerTile) *
      kScaleRowsPerTile;
  const std::uint64_t padded_columns =
      ((columns + kScaleColumnsPerTile - 1U) / kScaleColumnsPerTile) *
      kScaleColumnsPerTile;
  if (padded_rows >
      std::numeric_limits<std::uint64_t>::max() / padded_columns) {
    return Invalid("CUTLASS activation-scale interleave size overflow");
  }
  const std::uint64_t elements =
      padded_rows * padded_columns / kScaleColumnsPerTile;
  const unsigned blocks =
      static_cast<unsigned>((elements + kThreads - 1U) / kThreads);
  InterleaveActivationScalesKernel<<<blocks, kThreads, 0, stream>>>(
      compact_scales_e4m3fn, interleaved_scales_e4m3fn, tokens, columns,
      padded_rows, padded_columns);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch CUTLASS activation-scale interleave", error);
}

Status LaunchNvfp4CutlassProjectionBf16Batch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* interleaved_activation_scales_e4m3fn,
    const std::uint8_t* tiled_weight_e2m1,
    const std::uint8_t* tiled_weight_scales_e4m3fn,
    std::uint8_t* row_major_weight_scratch_e2m1,
    std::uint8_t* interleaved_weight_scale_scratch_e4m3fn,
    void* cutlass_workspace,
    std::size_t cutlass_workspace_bytes,
    std::uint16_t* output_bf16,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    float activation_global_divisor,
    float weight_global_divisor,
    cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr ||
      interleaved_activation_scales_e4m3fn == nullptr ||
      tiled_weight_e2m1 == nullptr ||
      tiled_weight_scales_e4m3fn == nullptr ||
      row_major_weight_scratch_e2m1 == nullptr ||
      interleaved_weight_scale_scratch_e4m3fn == nullptr ||
      output_bf16 == nullptr || tokens == 0U || rows == 0U ||
      contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(activation_global_divisor) ||
      !PositiveFinite(weight_global_divisor) ||
      tokens > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      contracting_elements >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Invalid("invalid CUTLASS NVFP4 projection arguments");
  }
  const std::uint64_t packed_copies =
      rows * contracting_elements / 2U / sizeof(uint4);
  const std::uint64_t scale_groups =
      rows * contracting_elements / kElementsPerKBlock;
  const std::uint64_t work = max(packed_copies, scale_groups);
  if (work > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
                 kThreads) {
    return Invalid("CUTLASS NVFP4 weight preparation grid is too large");
  }
  const unsigned blocks =
      static_cast<unsigned>((work + kThreads - 1U) / kThreads);
  PrepareWeightKernel<<<blocks, kThreads, 0, stream>>>(
      tiled_weight_e2m1, tiled_weight_scales_e4m3fn,
      row_major_weight_scratch_e2m1,
      interleaved_weight_scale_scratch_e4m3fn, rows,
      contracting_elements);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch CUTLASS NVFP4 weight preparation", error);
  }
  const float alpha =
      1.0F / (activation_global_divisor * weight_global_divisor);
  return LaunchGemm(
      packed_activation_e2m1, interleaved_activation_scales_e4m3fn,
      row_major_weight_scratch_e2m1,
      interleaved_weight_scale_scratch_e4m3fn, output_bf16,
      static_cast<int>(tokens), static_cast<int>(rows),
      static_cast<int>(contracting_elements), alpha, cutlass_workspace,
      cutlass_workspace_bytes, stream);
}

Status LaunchNvfp4CutlassUpGatedGeluQuantizedBatch(
    const std::uint8_t* packed_activation_e2m1,
    const std::uint8_t* interleaved_activation_scales_e4m3fn,
    const std::uint8_t* tiled_up_weight_e2m1,
    const std::uint8_t* tiled_up_weight_scales_e4m3fn,
    std::uint8_t* row_major_weight_scratch_e2m1,
    std::uint8_t* interleaved_weight_scale_scratch_e4m3fn,
    void* cutlass_workspace,
    std::size_t cutlass_workspace_bytes,
    const std::uint16_t* gate_bf16,
    std::uint8_t* product_packed_e2m1,
    std::uint8_t* product_interleaved_scales_e4m3fn,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    float up_activation_global_divisor,
    float up_weight_global_divisor,
    float product_global_divisor,
    const float* product_global_divisor_device,
    cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr ||
      interleaved_activation_scales_e4m3fn == nullptr ||
      tiled_up_weight_e2m1 == nullptr ||
      tiled_up_weight_scales_e4m3fn == nullptr ||
      row_major_weight_scratch_e2m1 == nullptr ||
      interleaved_weight_scale_scratch_e4m3fn == nullptr ||
      cutlass_workspace == nullptr || gate_bf16 == nullptr ||
      product_packed_e2m1 == nullptr ||
      product_interleaved_scales_e4m3fn == nullptr || tokens == 0U ||
      product_global_divisor_device == nullptr ||
      rows == 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U ||
      !PositiveFinite(up_activation_global_divisor) ||
      !PositiveFinite(up_weight_global_divisor) ||
      !PositiveFinite(product_global_divisor) ||
      tokens > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) -
                   (kScaleRowsPerTile - 1U) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      contracting_elements >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Invalid("invalid CUTLASS gated NVFP4 projection arguments");
  }
  const std::uint64_t packed_copies =
      rows * contracting_elements / 2U / sizeof(uint4);
  const std::uint64_t scale_groups =
      rows * contracting_elements / kElementsPerKBlock;
  const std::uint64_t work = max(packed_copies, scale_groups);
  if (work > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max()) *
                 kThreads) {
    return Invalid("CUTLASS gated NVFP4 weight preparation grid is too large");
  }
  const unsigned blocks =
      static_cast<unsigned>((work + kThreads - 1U) / kThreads);
  PrepareWeightKernel<<<blocks, kThreads, 0, stream>>>(
      tiled_up_weight_e2m1, tiled_up_weight_scales_e4m3fn,
      row_major_weight_scratch_e2m1,
      interleaved_weight_scale_scratch_e4m3fn, rows,
      contracting_elements);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaFailure("launch gated CUTLASS NVFP4 weight preparation",
                       error);
  }
  const float up_scale =
      1.0F /
      (up_activation_global_divisor * up_weight_global_divisor);
  // CUTLASS's block-scaled FP4 epilogue can write invalid packed payload
  // bytes in a partial final M tile. All production prefill arenas already
  // reserve the 128-row scale-tile extent, whose padded scale rows are zero.
  // Make that physical extent the GEMM problem so every observable row is in
  // a complete tile; the following Down GEMM still consumes only `tokens`.
  const std::uint64_t padded_tokens =
      ((tokens + kScaleRowsPerTile - 1U) / kScaleRowsPerTile) *
      kScaleRowsPerTile;
  return LaunchGatedGemm(
      packed_activation_e2m1, interleaved_activation_scales_e4m3fn,
      row_major_weight_scratch_e2m1,
      interleaved_weight_scale_scratch_e4m3fn, gate_bf16,
      product_packed_e2m1, product_interleaved_scales_e4m3fn,
      static_cast<int>(padded_tokens), static_cast<int>(rows),
      static_cast<int>(contracting_elements), up_scale,
      product_global_divisor_device, cutlass_workspace,
      cutlass_workspace_bytes,
      stream);
}

}  // namespace gem16::internal
