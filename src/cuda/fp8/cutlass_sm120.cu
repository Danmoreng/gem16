#include "cuda/fp8/cutlass_sm120.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#if defined(_WIN32)
#include "cuda/cutlass_windows_launch.cuh"
#endif

namespace gem16::internal {
namespace {

using namespace cute;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status Internal(std::string message) {
  return Status(StatusCode::kInternal, std::move(message));
}

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ThreadBlockShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

constexpr auto kRoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

// Preserve the checkpoint's exact left-to-right scaling and BF16 projection
// boundary inside the CUTLASS epilogue:
//
//   BF16((acc * activation_scale[token]) * weight_scale[row])
using ScaledBf16Epilogue = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementD, float, kRoundStyle>,
    cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, ThreadBlockShape, cutlass::bfloat16_t, float>,
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::multiplies, float, float, kRoundStyle>,
        cutlass::epilogue::fusion::Sm90ColBroadcast<
            0, ThreadBlockShape, float, float>,
        cutlass::epilogue::fusion::Sm90AccFetch>>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator, void,
        cutlass::layout::RowMajor, 1, ElementD,
        cutlass::layout::RowMajor, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        ScaledBf16Epilogue>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, cutlass::layout::RowMajor, 16,
        ElementB, cutlass::layout::ColumnMajor, 16, ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
    cutlass::gemm::StaticPersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

Status LaunchGemm(
    const std::uint8_t* activation, const float* activation_scales,
    const std::uint8_t* weight, const std::uint16_t* weight_scales,
    std::uint16_t* output, int m, int n, int k, void* workspace,
    std::size_t workspace_bytes, cudaStream_t stream) {
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
      {reinterpret_cast<const ElementA*>(activation), stride_a,
       reinterpret_cast<const ElementB*>(weight), stride_b},
      {{}, nullptr, stride_d, reinterpret_cast<ElementD*>(output), stride_d}};
  arguments.epilogue.thread = {
      {{reinterpret_cast<const cutlass::bfloat16_t*>(weight_scales)}},
      {{{activation_scales}}, {}, {}},
      {}};

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
    return Invalid("CUTLASS FP8 workspace is too small");
  }
#if defined(_WIN32)
  cutlass::Status status = cutlass_windows::InitializeAndRun<Gemm>(
      arguments, workspace, workspace_bytes, stream);
  if (status != cutlass::Status::kSuccess) {
    return Internal(std::string("CUTLASS FP8 Windows launch failed: ") +
                    cutlass::cutlassGetStatusString(status));
  }
#else
  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return Invalid(std::string("CUTLASS cannot implement FP8 projection: ") +
                   cutlass::cutlassGetStatusString(status));
  }
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return Internal(std::string("CUTLASS FP8 initialization failed: ") +
                    cutlass::cutlassGetStatusString(status));
  }
  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return Internal(std::string("CUTLASS FP8 launch failed: ") +
                    cutlass::cutlassGetStatusString(status));
  }
#endif
  return Status::Ok();
}

}  // namespace

Status LaunchFp8CutlassProjectionBatch(
    const std::uint8_t* activation_e4m3fn,
    const float* activation_scales,
    const std::uint8_t* weight_e4m3fn,
    const std::uint16_t* weight_scales_bf16,
    std::uint16_t* output_bf16,
    std::uint64_t tokens,
    std::uint64_t rows,
    std::uint64_t contracting_elements,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream) {
  if (activation_e4m3fn == nullptr || activation_scales == nullptr ||
      weight_e4m3fn == nullptr || weight_scales_bf16 == nullptr ||
      output_bf16 == nullptr || workspace == nullptr || tokens == 0U ||
      rows == 0U || contracting_elements == 0U ||
      tokens > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      contracting_elements >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Invalid("invalid CUTLASS FP8 projection arguments");
  }
  return LaunchGemm(
      activation_e4m3fn, activation_scales, weight_e4m3fn,
      weight_scales_bf16, output_bf16, static_cast<int>(tokens),
      static_cast<int>(rows), static_cast<int>(contracting_elements), workspace,
      workspace_bytes, stream);
}

}  // namespace gem16::internal
