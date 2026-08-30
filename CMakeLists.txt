cmake_minimum_required(VERSION 3.28)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" GEM16_VERSION LIMIT_COUNT 1)
string(STRIP "${GEM16_VERSION}" GEM16_VERSION)
if(NOT GEM16_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(-[0-9A-Za-z.-]+)?(\\+[0-9A-Za-z.-]+)?$")
  message(FATAL_ERROR "VERSION must contain one semantic version, got '${GEM16_VERSION}'")
endif()
set(GEM16_CMAKE_VERSION
    "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")

project(gem16 VERSION "${GEM16_CMAKE_VERSION}" LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(GEM16_ENABLE_CUDA "Build the CUDA capability path" OFF)
option(GEM16_BUILD_TESTS "Build host tests" ON)
option(GEM16_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(GEM16_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)
set(GEM16_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass" CACHE PATH
    "Path to the pinned CUTLASS source tree")

include(CompilerWarnings)
include(Sanitizers)
include(CudaArchitectures)

# JSON parsing is shared by the runtime and the host-only offline compiler.
# Keep it in a CUDA-free archive so the compiler target never pulls the
# CUDA-bearing core archive into its link.
add_library(gem16_json STATIC
  src/util/json.cpp
)
target_include_directories(gem16_json
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_features(gem16_json PUBLIC cxx_std_20)
gem16_set_project_warnings(gem16_json)
gem16_enable_sanitizers(gem16_json)
# GCC 16 emits a false positive from std::variant's move constructor at -O3.
# Keep the suppression local to the JSON translation unit.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_property(SOURCE src/util/json.cpp APPEND PROPERTY COMPILE_OPTIONS -Wno-maybe-uninitialized)
  set_property(SOURCE src/model/gemma4_26b_compiled_loader.cpp APPEND PROPERTY COMPILE_OPTIONS -Wno-maybe-uninitialized)
  set_property(SOURCE src/model/gemma4_26b_trellis35.cpp APPEND PROPERTY COMPILE_OPTIONS -Wno-maybe-uninitialized)
endif()

add_library(gem16_core STATIC
  src/numeric/fp8.cpp
  src/numeric/layer.cpp
  src/numeric/nvfp4.cpp
  src/numeric/nvfp4_head.cpp
  src/cuda/nvfp4/sm120_layout.cpp
  src/cuda/moe/prefill_plan.cpp
  src/model/checkpoint_loader.cpp
  src/model/gemma4_26b_attention.cpp
  src/model/gemma4_26b_compiled_loader.cpp
  src/model/gemma4_26b_device_image.cpp
  src/model/gemma4_26b_residency.cpp
  src/model/gemma4_26b_trellis35.cpp
  src/model/audio.cpp
  src/model/image.cpp
  src/model/config.cpp
  src/model/gemma4_26b_manifest.cpp
  src/model/manifest.cpp
  src/model/model_variant.cpp
  src/model/safetensors.cpp
  src/model/tokenizer.cpp
  src/model/tokenizer_config.cpp
  src/runtime/capabilities.cpp
  src/runtime/chat.cpp
  src/runtime/tool_call_parser.cpp
  src/runtime/memory.cpp
  src/runtime/memory_plan.cpp
  src/runtime/sampling.cpp
  src/runtime/result_json.cpp
  src/cuda/engine/state_capture.cpp
  src/compiler/sha256.cpp
)
if(WIN32)
  target_sources(gem16_core PRIVATE src/platform/mapped_file_windows.cpp)
else()
  target_sources(gem16_core PRIVATE src/platform/mapped_file_posix.cpp)
endif()
target_include_directories(gem16_core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_include_directories(gem16_core SYSTEM PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/miniaudio"
)
target_compile_features(gem16_core PUBLIC cxx_std_20)
target_link_libraries(gem16_core PRIVATE gem16_json)

# Keep full device-image integrity validation while using the platform's
# hardware-accelerated SHA-256 implementation when it is available.
find_package(OpenSSL QUIET COMPONENTS Crypto)
if(OpenSSL_FOUND)
  target_link_libraries(gem16_core PRIVATE OpenSSL::Crypto)
  target_compile_definitions(gem16_core PRIVATE GEM16_HAS_OPENSSL=1)
endif()
gem16_set_project_warnings(gem16_core)
gem16_enable_sanitizers(gem16_core)

find_package(Threads REQUIRED)

# Bind native conversion data planes to the exact configured toolchain and target.
set(GEM16_NATIVE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
if(GEM16_NATIVE_BUILD_TYPE STREQUAL "")
  set(GEM16_NATIVE_BUILD_TYPE "multi-config")
endif()
set(GEM16_NATIVE_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(GEM16_NATIVE_PROCESSOR STREQUAL "")
  set(GEM16_NATIVE_PROCESSOR "unknown")
endif()

add_library(gem16_fp8_compiler_support STATIC
  src/compiler/fp8_batch_encoder.cpp
  src/compiler/sha256.cpp
)
target_include_directories(gem16_fp8_compiler_support
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(gem16_fp8_compiler_support PRIVATE gem16_json Threads::Threads)
target_compile_definitions(gem16_fp8_compiler_support PRIVATE
  GEM16_FP8_NATIVE_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  GEM16_FP8_NATIVE_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
  GEM16_FP8_NATIVE_BUILD_TYPE="${GEM16_NATIVE_BUILD_TYPE}"
  GEM16_FP8_NATIVE_CXX_STANDARD="20"
  GEM16_FP8_NATIVE_SYSTEM="${CMAKE_SYSTEM_NAME}"
  GEM16_FP8_NATIVE_PROCESSOR="${GEM16_NATIVE_PROCESSOR}"
)
if(MSVC)
  target_compile_options(gem16_fp8_compiler_support PRIVATE /fp:strict)
else()
  target_compile_options(gem16_fp8_compiler_support PRIVATE -fno-fast-math -ffp-contract=off)
endif()
gem16_set_project_warnings(gem16_fp8_compiler_support)
gem16_enable_sanitizers(gem16_fp8_compiler_support)

add_executable(gem16-fp8-compiler src/cli/fp8_compiler_main.cpp)
target_include_directories(gem16-fp8-compiler PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(gem16-fp8-compiler PRIVATE gem16_fp8_compiler_support)
gem16_set_project_warnings(gem16-fp8-compiler)
gem16_enable_sanitizers(gem16-fp8-compiler)

add_library(gem16_nvfp4_compiler_support STATIC
  src/compiler/nvfp4_batch_encoder.cpp
  src/compiler/sha256.cpp
)
target_include_directories(gem16_nvfp4_compiler_support
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(gem16_nvfp4_compiler_support PRIVATE gem16_json Threads::Threads)
target_compile_definitions(gem16_nvfp4_compiler_support PRIVATE
  GEM16_NATIVE_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  GEM16_NATIVE_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
  GEM16_NATIVE_BUILD_TYPE="${GEM16_NATIVE_BUILD_TYPE}"
  GEM16_NATIVE_CXX_STANDARD="20"
  GEM16_NATIVE_SYSTEM="${CMAKE_SYSTEM_NAME}"
  GEM16_NATIVE_PROCESSOR="${GEM16_NATIVE_PROCESSOR}"
)
if(MSVC)
  target_compile_options(gem16_nvfp4_compiler_support PRIVATE /fp:strict)
else()
  target_compile_options(gem16_nvfp4_compiler_support PRIVATE -fno-fast-math -ffp-contract=off)
endif()
gem16_set_project_warnings(gem16_nvfp4_compiler_support)
gem16_enable_sanitizers(gem16_nvfp4_compiler_support)

add_executable(gem16-checkpoint-compiler src/cli/checkpoint_compiler_main.cpp)
target_include_directories(gem16-checkpoint-compiler PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(gem16-checkpoint-compiler PRIVATE gem16_nvfp4_compiler_support)
gem16_set_project_warnings(gem16-checkpoint-compiler)
gem16_enable_sanitizers(gem16-checkpoint-compiler)

add_executable(gem16-nvfp4-head-diagnostic
  src/cli/nvfp4_head_diagnostic_main.cpp
)
target_include_directories(gem16-nvfp4-head-diagnostic PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(gem16-nvfp4-head-diagnostic PRIVATE gem16_core gem16_json)
gem16_set_project_warnings(gem16-nvfp4-head-diagnostic)
gem16_enable_sanitizers(gem16-nvfp4-head-diagnostic)

add_library(gem16_server_support STATIC
  src/server/http_streaming.cpp
  src/server/observability.cpp
  src/server/openai_chat.cpp
  src/server/request_queue.cpp
  src/server/secure_id.cpp
  src/server/session_pool.cpp
)
target_include_directories(gem16_server_support
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_include_directories(gem16_server_support SYSTEM PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpp-httplib"
)
target_link_libraries(gem16_server_support PUBLIC gem16_core)
if(WIN32)
  target_link_libraries(gem16_server_support PRIVATE bcrypt)
endif()
gem16_set_project_warnings(gem16_server_support)
gem16_enable_sanitizers(gem16_server_support)
if(GEM16_ENABLE_CUDA)
  # This C++-only archive forwards gem16_core to final executables. Deferring
  # CUDA symbol resolution avoids consuming core registration objects in an
  # intermediate device link before gem16-server is linked.
  set_target_properties(gem16_server_support PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  )
endif()

add_library(gem16_httplib STATIC
  third_party/cpp-httplib/httplib.cpp
)
target_include_directories(gem16_httplib SYSTEM PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpp-httplib"
)
target_compile_definitions(gem16_httplib PRIVATE
  CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=16777216
  CPPHTTPLIB_TCP_NODELAY=1
)
target_link_libraries(gem16_httplib PRIVATE Threads::Threads)
if(WIN32)
  target_link_libraries(gem16_httplib PUBLIC ws2_32)
endif()
if(MSVC)
  target_compile_options(gem16_httplib PRIVATE /w)
else()
  target_compile_options(gem16_httplib PRIVATE -w)
endif()

if(GEM16_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "GEM16_ENABLE_CUDA=ON, but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  gem16_configure_cuda_architectures()
  add_executable(gem16-trellis35-quantize
    src/cli/trellis35_quantize_main.cu
  )
  target_include_directories(gem16-trellis35-quantize PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(gem16-trellis35-quantize PRIVATE CUDA::cudart)
  set_target_properties(gem16-trellis35-quantize PROPERTIES
    CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
  )
  gem16_set_project_warnings(gem16-trellis35-quantize)
  add_executable(gem16-trellis35-calibration
    src/cli/trellis35_calibration_main.cpp
  )
  target_include_directories(gem16-trellis35-calibration PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-trellis35-calibration PRIVATE
    gem16_core CUDA::cudart
  )
  set_target_properties(gem16-trellis35-calibration PROPERTIES
    CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
  )
  gem16_set_project_warnings(gem16-trellis35-calibration)
  add_executable(gem16-trellis35-expand-down
    src/cli/trellis35_expand_down_main.cu
  )
  target_link_libraries(gem16-trellis35-expand-down PRIVATE
    CUDA::cudart CUDA::cublas
  )
  set_target_properties(gem16-trellis35-expand-down PROPERTIES
    CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
  )
  gem16_set_project_warnings(gem16-trellis35-expand-down)
  add_executable(gem16-trellis35-ldlq
    src/cli/trellis35_ldlq_main.cu
  )
  target_include_directories(gem16-trellis35-ldlq PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(gem16-trellis35-ldlq PRIVATE
    CUDA::cudart CUDA::cublas
  )
  set_target_properties(gem16-trellis35-ldlq PROPERTIES
    CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
  )
  gem16_set_project_warnings(gem16-trellis35-ldlq)
  add_executable(gem16-trellis35-load-probe
    src/cli/trellis35_load_probe.cpp
  )
  target_include_directories(gem16-trellis35-load-probe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-trellis35-load-probe PRIVATE
    gem16_core CUDA::cudart
  )
  set_target_properties(gem16-trellis35-load-probe PROPERTIES
    CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
  )
  gem16_set_project_warnings(gem16-trellis35-load-probe)
  target_sources(gem16_core PRIVATE
    src/cuda/attention/decode_sm120.cu
    src/cuda/attention/gemma4_26b_reference.cu
    src/cuda/attention/prefill_local_sm120.cu
    src/cuda/attention/prefill_global_sm120.cu
    src/cuda/attention/reference.cu
    src/cuda/audio/projection.cu
    src/cuda/dispatch.cu
    src/cuda/engine/gemma4_26b_artifact.cpp
    src/cuda/engine/gemma4_26b_trellis35_artifact.cpp
    src/cuda/engine/gemma4_26b_reference.cu
    src/cuda/engine/target_model.cu
    src/cuda/engine/inference_engine.cu
    src/cuda/inference.cu
    src/cuda/fp8/checkpoint_probe.cu
    src/cuda/fp8/cutlass_sm120.cu
    src/cuda/fp8/reference.cu
    src/cuda/fp8/sm120.cu
    src/cuda/kv_cache/reference.cu
    src/cuda/norm/reference.cu
    src/cuda/rope/reference.cu
    src/cuda/layer/checkpoint_probe.cu
    src/cuda/mtp/assistant.cu
    src/cuda/mtp/gemma4_26b_assistant.cu
    src/cuda/mtp/verify.cu
    src/cuda/moe/gemma4_26b_bindings.cpp
    src/cuda/moe/prefill.cu
    src/cuda/moe/reference.cu
    src/cuda/moe/router_diagnostic.cu
    src/cuda/nvfp4/checkpoint_probe.cu
    src/cuda/nvfp4/gemv.cu
    src/cuda/nvfp4/mlp.cu
    src/cuda/nvfp4/mlp_checkpoint_probe.cu
    src/cuda/nvfp4/reference.cu
    src/cuda/nvfp4/sm120.cu
    src/cuda/nvfp4/cutlass_sm120.cu
    src/cuda/output_head.cu
    src/cuda/sampling/sampling.cu
    src/cuda/trellis35/reference.cu
    src/cuda/vision/projection.cu
  )
  set_property(SOURCE src/cuda/trellis35/reference.cu APPEND PROPERTY
    INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
  )
  if(NOT EXISTS "${GEM16_CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
      "SM120 CUDA builds require CUTLASS headers at GEM16_CUTLASS_DIR=${GEM16_CUTLASS_DIR}")
  endif()
  target_include_directories(gem16_core SYSTEM PRIVATE
    "${GEM16_CUTLASS_DIR}/include"
    "${GEM16_CUTLASS_DIR}/tools/util/include"
  )
  if(MSVC)
    # CUTLASS 4.x's CuTe templates do not parse under the strict conformance
    # mode that MSVC enables implicitly for C++20. Keep the documented
    # preprocessor requirements and permissive parser mode local to the two
    # translation units that instantiate CUTLASS kernels.
    set_property(SOURCE
      src/cuda/fp8/cutlass_sm120.cu
      src/cuda/nvfp4/cutlass_sm120.cu
      APPEND PROPERTY COMPILE_OPTIONS
        -Xcompiler=/Zc:preprocessor
        -Xcompiler=/Zc:__cplusplus
        -Xcompiler=/bigobj
        -Xcompiler=/permissive
    )
    # CUDA 13.3's cuda_fp4.hpp and nvcc's generated registration stub trigger these host warnings.
    # Keep the suppression on the one translation unit that includes the FP4 intrinsic header.
    set_property(SOURCE src/cuda/nvfp4/reference.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4100
      -Xcompiler=/wd4211
    )
    set_property(SOURCE src/cuda/fp8/reference.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4100
      -Xcompiler=/wd4211
    )
    set_property(SOURCE src/cuda/fp8/checkpoint_probe.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4211
      -diag-suppress=2810
    )
    set_property(SOURCE src/cuda/fp8/sm120.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4211
    )
    set_property(SOURCE
      src/cuda/attention/decode_sm120.cu
      src/cuda/attention/prefill_local_sm120.cu
      src/cuda/attention/prefill_global_sm120.cu
    src/cuda/attention/reference.cu
      APPEND PROPERTY COMPILE_OPTIONS
        -Xcompiler=/wd4211
    )
    set_property(SOURCE
      src/cuda/attention/reference.cu
      src/cuda/kv_cache/reference.cu
      src/cuda/norm/reference.cu
      src/cuda/rope/reference.cu
      APPEND PROPERTY COMPILE_OPTIONS
        -Xcompiler=/wd4211
    )
    set_property(SOURCE src/cuda/layer/checkpoint_probe.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4211
      -diag-suppress=2810
    )
    set_property(SOURCE src/cuda/nvfp4/sm120.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4211
    )
    set_property(SOURCE src/cuda/nvfp4/gemv.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4211
    )
    set_property(SOURCE src/cuda/nvfp4/mlp.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4211
    )
    set_property(SOURCE src/cuda/nvfp4/mlp_checkpoint_probe.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4211
      -diag-suppress=2810
    )
    set_property(SOURCE src/cuda/nvfp4/checkpoint_probe.cu APPEND PROPERTY COMPILE_OPTIONS
      -diag-suppress=2810
    )
    # inference.cu contains internal-linkage kernels, which make CUDA 13.3's
    # generated registration stub trigger C4211 under MSVC. NVCC also emits
    # 2810 for Status results that are assigned and checked in this file.
    # Keep both compiler-specific suppressions local to this translation unit.
    set_property(SOURCE src/cuda/inference.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4211
      -diag-suppress=2810
    )
    set_property(SOURCE src/cuda/sampling/sampling.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4211
      # CUB/CCCL rejects MSVC's traditional preprocessor under CUDA 13.3.
      -Xcompiler=/Zc:preprocessor
    )
  endif()
  target_link_libraries(gem16_core PRIVATE CUDA::cudart CUDA::nvtx3)
  if(UNIX AND NOT APPLE AND TARGET CUDA::cuFile)
    target_link_libraries(gem16_core PRIVATE CUDA::cuFile)
    target_compile_definitions(gem16_core PRIVATE GEM16_HAS_CUFILE=1)
  endif()
  target_compile_definitions(gem16_core
    PUBLIC GEM16_HAS_CUDA=1
    PRIVATE GEM16_COMPILED_CUDA_ARCH="${GEM16_CUDA_ARCHITECTURE}"
  )
  set_target_properties(gem16_core PROPERTIES
    CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
  )
  set_property(SOURCE src/cuda/engine/gemma4_26b_reference.cu
    APPEND PROPERTY COMPILE_OPTIONS -diag-suppress=2810)
  set_property(SOURCE src/cuda/mtp/gemma4_26b_assistant.cu
    APPEND PROPERTY COMPILE_OPTIONS -diag-suppress=2810)
  if(MSVC)
    set_property(SOURCE src/cuda/engine/gemma4_26b_reference.cu
      APPEND PROPERTY COMPILE_OPTIONS -Xcompiler=/wd4211)
  endif()
  add_executable(gem16-26b-memory-probe
    src/cli/gemma4_26b_memory_probe.cpp
  )
  target_include_directories(gem16-26b-memory-probe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-memory-probe PRIVATE gem16_core CUDA::cudart)
  gem16_set_project_warnings(gem16-26b-memory-probe)
  gem16_enable_sanitizers(gem16-26b-memory-probe)
  add_executable(gem16-26b-assistant-residency-probe
    src/cli/gemma4_26b_assistant_residency_probe.cpp
  )
  target_include_directories(gem16-26b-assistant-residency-probe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-assistant-residency-probe PRIVATE
    gem16_core CUDA::cudart
  )
  gem16_set_project_warnings(gem16-26b-assistant-residency-probe)
  gem16_enable_sanitizers(gem16-26b-assistant-residency-probe)
  add_executable(gem16-26b-assistant-smoke
    src/cli/gemma4_26b_assistant_smoke.cpp
  )
  target_include_directories(gem16-26b-assistant-smoke PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-assistant-smoke PRIVATE
    gem16_core CUDA::cudart
  )
  gem16_set_project_warnings(gem16-26b-assistant-smoke)
  gem16_enable_sanitizers(gem16-26b-assistant-smoke)
  add_executable(gem16-26b-assistant-acceptance
    src/cli/gemma4_26b_assistant_acceptance.cpp
  )
  target_include_directories(gem16-26b-assistant-acceptance PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-assistant-acceptance PRIVATE
    gem16_core CUDA::cudart
  )
  gem16_set_project_warnings(gem16-26b-assistant-acceptance)
  gem16_enable_sanitizers(gem16-26b-assistant-acceptance)
  add_executable(gem16-26b-moe-probe
    src/cli/gemma4_26b_moe_probe.cpp
  )
  target_include_directories(gem16-26b-moe-probe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-moe-probe PRIVATE gem16_core CUDA::cudart)
  gem16_set_project_warnings(gem16-26b-moe-probe)
  gem16_enable_sanitizers(gem16-26b-moe-probe)
  add_executable(gem16-26b-router-diagnostic
    src/cli/gemma4_26b_router_diagnostic.cpp
  )
  target_include_directories(gem16-26b-router-diagnostic PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-router-diagnostic PRIVATE
    gem16_core CUDA::cudart
  )
  gem16_set_project_warnings(gem16-26b-router-diagnostic)
  gem16_enable_sanitizers(gem16-26b-router-diagnostic)
  add_executable(gem16-26b-attention-probe
    src/cli/gemma4_26b_attention_probe.cpp
  )
  target_include_directories(gem16-26b-attention-probe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-attention-probe PRIVATE gem16_core CUDA::cudart)
  gem16_set_project_warnings(gem16-26b-attention-probe)
  gem16_enable_sanitizers(gem16-26b-attention-probe)
  add_executable(gem16-26b-reference
    src/cli/gemma4_26b_reference_main.cpp
  )
  target_include_directories(gem16-26b-reference PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-reference PRIVATE gem16_core CUDA::cudart)
  gem16_set_project_warnings(gem16-26b-reference)
  gem16_enable_sanitizers(gem16-26b-reference)
  add_executable(gem16-26b-benchmark
    src/cli/gemma4_26b_benchmark_main.cpp
  )
  target_include_directories(gem16-26b-benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-benchmark PRIVATE
    gem16_core CUDA::cudart
  )
  gem16_set_project_warnings(gem16-26b-benchmark)
  gem16_enable_sanitizers(gem16-26b-benchmark)
  add_executable(gem16-26b-m21-context-driver
    tests/cuda/gemma4_26b_m21_context_driver.cpp
  )
  target_include_directories(gem16-26b-m21-context-driver PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-m21-context-driver PRIVATE
    gem16_core CUDA::cudart
  )
  set_target_properties(gem16-26b-m21-context-driver PROPERTIES
    CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
  )
  gem16_set_project_warnings(gem16-26b-m21-context-driver)
  gem16_enable_sanitizers(gem16-26b-m21-context-driver)
  add_executable(gem16-26b-m25-context-driver
    tests/cuda/gemma4_26b_m25_context_driver.cpp
  )
  target_include_directories(gem16-26b-m25-context-driver PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-m25-context-driver PRIVATE
    gem16_core CUDA::cudart
  )
  set_target_properties(gem16-26b-m25-context-driver PROPERTIES
    CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
  )
  gem16_set_project_warnings(gem16-26b-m25-context-driver)
  gem16_enable_sanitizers(gem16-26b-m25-context-driver)
  add_executable(gem16-26b-m22-product-driver
    tests/cuda/gemma4_26b_m22_product_driver.cpp
  )
  target_include_directories(gem16-26b-m22-product-driver PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-m22-product-driver PRIVATE
    gem16_core CUDA::cudart
  )
  set_target_properties(gem16-26b-m22-product-driver PROPERTIES
    CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
  )
  gem16_set_project_warnings(gem16-26b-m22-product-driver)
  gem16_enable_sanitizers(gem16-26b-m22-product-driver)
  add_executable(gem16-26b-m19-q4-driver
    tests/cuda/gemma4_26b_m19_q4_driver.cpp
  )
  target_include_directories(gem16-26b-m19-q4-driver PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(gem16-26b-m19-q4-driver PRIVATE
    gem16_core CUDA::cudart
  )
  set_target_properties(gem16-26b-m19-q4-driver PROPERTIES
    CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
  )
  gem16_set_project_warnings(gem16-26b-m19-q4-driver)
  gem16_enable_sanitizers(gem16-26b-m19-q4-driver)
  message(STATUS "gem16 CUDA capability scaffold: enabled for ${CMAKE_CUDA_ARCHITECTURES}")
else()
  target_sources(gem16_core PRIVATE src/runtime/inference_stub.cpp)
  target_compile_definitions(gem16_core PUBLIC GEM16_HAS_CUDA=0)
  message(STATUS "gem16 CUDA capability scaffold: disabled")
endif()

foreach(tool IN ITEMS inspect run bench chat)
  add_executable(gem16-${tool} src/cli/${tool}_main.cpp)
  target_link_libraries(gem16-${tool} PRIVATE gem16_core)
  gem16_set_project_warnings(gem16-${tool})
  gem16_enable_sanitizers(gem16-${tool})
endforeach()
target_include_directories(gem16-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(gem16-chat PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_executable(gem16-server src/cli/server_main.cpp)
target_include_directories(gem16-server PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(gem16-server PRIVATE
  GEM16_VERSION_STRING="${GEM16_VERSION}"
)
# Keep the CUDA-bearing core as a direct dependency so CMake includes its
# separable-compilation objects in the server executable's device-link step.
target_link_libraries(gem16-server PRIVATE gem16_server_support gem16_httplib gem16_core)
gem16_set_project_warnings(gem16-server)
gem16_enable_sanitizers(gem16-server)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME gem16-server-version COMMAND gem16-server --version)
  set_tests_properties(gem16-server-version PROPERTIES
    PASS_REGULAR_EXPRESSION "gem16-server ${GEM16_VERSION}"
  )
endif()
if(BUILD_TESTING AND GEM16_BUILD_TESTS)
  add_executable(gem16-unit-tests
    tests/unit/audio_test.cpp
    tests/unit/chat_test.cpp
    tests/unit/config_test.cpp
    tests/unit/fp8_test.cpp
    tests/unit/fp8_batch_encoder_test.cpp
    tests/unit/gemma4_26b_manifest_test.cpp
    tests/unit/gemma4_26b_trellis35_test.cpp
    tests/unit/image_test.cpp
    tests/unit/json_test.cpp
    tests/unit/gemma4_26b_moe_prefill_plan_test.cpp
    tests/unit/layer_test.cpp
    tests/unit/memory_plan_test.cpp
    tests/unit/mtp_scheduler_test.cpp
    tests/unit/nvfp4_test.cpp
    tests/unit/nvfp4_batch_encoder_test.cpp
    tests/unit/nvfp4_head_test.cpp
    tests/unit/openai_chat_test.cpp
    tests/unit/observability_test.cpp
    tests/unit/request_queue_test.cpp
    tests/unit/safetensors_test.cpp
    tests/unit/sampling_test.cpp
    tests/unit/sm120_layout_test.cpp
    tests/unit/tokenizer_test.cpp
    tests/unit/test_main.cpp
  )
  if(WIN32)
    target_sources(gem16-unit-tests PRIVATE tests/unit/windows_utf8_test.cpp)
  endif()
  target_include_directories(gem16-unit-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit")
  target_link_libraries(gem16-unit-tests PRIVATE gem16_core gem16_server_support gem16_fp8_compiler_support gem16_nvfp4_compiler_support)
  gem16_set_project_warnings(gem16-unit-tests)
  gem16_enable_sanitizers(gem16-unit-tests)
  add_test(NAME gem16-unit COMMAND gem16-unit-tests)

  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME gem16-26b-m23-host-contract
      COMMAND "${Python3_EXECUTABLE}" -m unittest
        tests.python.test_qualify_gemma4_26b_m23
    )
    set_tests_properties(gem16-26b-m23-host-contract PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME gem16-trellis35-compiler-contract
      COMMAND "${Python3_EXECUTABLE}" -m unittest
        tests.python.test_trellis35_compiler
        tests.python.test_trellis35_checkpoint
        tests.python.test_compare_gemma4_26b_wp8
        tests.python.test_benchmark_gemma4_26b_wp8_text
    )
    set_tests_properties(gem16-trellis35-compiler-contract PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()

  if(GEM16_ENABLE_CUDA)
    add_executable(gem16-cuda-tests tests/cuda/nvfp4_reference_test.cu)
    target_include_directories(gem16-cuda-tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(gem16-cuda-tests PRIVATE gem16_core CUDA::cudart)
    set_target_properties(gem16-cuda-tests PROPERTIES
      CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
    )
    gem16_set_project_warnings(gem16-cuda-tests)
    add_test(NAME gem16-cuda COMMAND gem16-cuda-tests)
    set_tests_properties(gem16-cuda PROPERTIES
      RESOURCE_LOCK gem16_gpu0
    )

    add_executable(gem16-cuda-moe-reference-tests
      tests/cuda/gemma4_26b_moe_reference_test.cu
    )
    target_include_directories(gem16-cuda-moe-reference-tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(gem16-cuda-moe-reference-tests PRIVATE
      gem16_core CUDA::cudart
    )
    set_target_properties(gem16-cuda-moe-reference-tests PROPERTIES
      CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
    )
    gem16_set_project_warnings(gem16-cuda-moe-reference-tests)
    add_test(NAME gem16-cuda-moe-reference
      COMMAND gem16-cuda-moe-reference-tests
    )
    set_tests_properties(gem16-cuda-moe-reference PROPERTIES
      RESOURCE_LOCK gem16_gpu0
    )

    add_executable(gem16-cuda-trellis35-tests
      tests/cuda/trellis35_m1_test.cu
    )
    target_include_directories(gem16-cuda-trellis35-tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(gem16-cuda-trellis35-tests PRIVATE
      gem16_core CUDA::cudart
    )
    set_target_properties(gem16-cuda-trellis35-tests PROPERTIES
      CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
    )
    gem16_set_project_warnings(gem16-cuda-trellis35-tests)
    add_test(NAME gem16-cuda-trellis35
      COMMAND gem16-cuda-trellis35-tests
    )
    set_tests_properties(gem16-cuda-trellis35 PROPERTIES
      RESOURCE_LOCK gem16_gpu0
    )

    add_executable(gem16-cuda-attention-reference-tests
      tests/cuda/gemma4_26b_attention_reference_test.cu
    )
    target_include_directories(gem16-cuda-attention-reference-tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(gem16-cuda-attention-reference-tests PRIVATE
      gem16_core CUDA::cudart
    )
    set_target_properties(gem16-cuda-attention-reference-tests PROPERTIES
      CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
    )
    gem16_set_project_warnings(gem16-cuda-attention-reference-tests)
    add_test(NAME gem16-cuda-attention-reference
      COMMAND gem16-cuda-attention-reference-tests
    )
    set_tests_properties(gem16-cuda-attention-reference PROPERTIES
      RESOURCE_LOCK gem16_gpu0
    )

    add_executable(gem16-cuda-nvfp4-consumption-tests
      tests/cuda/gemma4_26b_nvfp4_consumption_test.cu
    )
    target_include_directories(gem16-cuda-nvfp4-consumption-tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(gem16-cuda-nvfp4-consumption-tests PRIVATE
      gem16_core gem16_nvfp4_compiler_support CUDA::cudart
    )
    set_target_properties(gem16-cuda-nvfp4-consumption-tests PROPERTIES
      CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
    )
    gem16_set_project_warnings(gem16-cuda-nvfp4-consumption-tests)
    add_test(NAME gem16-cuda-nvfp4-consumption
      COMMAND gem16-cuda-nvfp4-consumption-tests
    )
    set_tests_properties(gem16-cuda-nvfp4-consumption PROPERTIES
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK gem16_gpu0
    )

    if(Python3_Interpreter_FOUND)
      add_test(NAME gem16-26b-m21-host-contract
        COMMAND "${Python3_EXECUTABLE}" -m unittest
          tests.python.test_qualify_gemma4_26b_m21
      )
      set_tests_properties(gem16-26b-m21-host-contract PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
      add_test(NAME gem16-26b-m17-engine-smoke
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_gemma4_26b_m17_engine_smoke.py"
          "$<TARGET_FILE:gem16-26b-reference>"
      )
      set_tests_properties(gem16-26b-m17-engine-smoke PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        RESOURCE_LOCK gem16_gpu0
      )
      add_test(NAME gem16-26b-m22-product
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_gemma4_26b_m22_product.py"
          26b
          "$<TARGET_FILE:gem16-26b-m22-product-driver>"
          "$<TARGET_FILE:gem16-chat>"
          "$<TARGET_FILE:gem16-server>"
      )
      set_tests_properties(gem16-26b-m22-product PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 1200
        RESOURCE_LOCK gem16_gpu0
      )
      add_test(NAME gem16-12b-m22-product
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_gemma4_26b_m22_product.py"
          12b
          "$<TARGET_FILE:gem16-run>"
          "$<TARGET_FILE:gem16-server>"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/vllm-gemma4-12b-nvfp4.json"
      )
      set_tests_properties(gem16-12b-m22-product PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 1200
        RESOURCE_LOCK gem16_gpu0
      )
    endif()
  endif()
endif()
