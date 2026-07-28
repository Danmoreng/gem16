cmake_minimum_required(VERSION 3.28)

project(gem16 VERSION 0.1.0 LANGUAGES CXX)

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

add_library(gem16_core STATIC
  src/numeric/fp8.cpp
  src/numeric/layer.cpp
  src/numeric/nvfp4.cpp
  src/cuda/nvfp4/sm120_layout.cpp
  src/model/checkpoint_loader.cpp
  src/model/config.cpp
  src/model/manifest.cpp
  src/model/safetensors.cpp
  src/model/tokenizer.cpp
  src/model/tokenizer_config.cpp
  src/runtime/capabilities.cpp
  src/runtime/memory.cpp
  src/runtime/memory_plan.cpp
  src/runtime/sampling.cpp
  src/runtime/result_json.cpp
  src/cuda/engine/state_capture.cpp
  src/util/json.cpp
)
if(WIN32)
  target_sources(gem16_core PRIVATE src/platform/mapped_file_windows.cpp)
else()
  target_sources(gem16_core PRIVATE src/platform/mapped_file_posix.cpp)
endif()
# GCC 16 emits a false positive from std::variant's move constructor at -O3.
# Keep the suppression local to the JSON translation unit.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_property(SOURCE src/util/json.cpp APPEND PROPERTY COMPILE_OPTIONS -Wno-maybe-uninitialized)
endif()
target_include_directories(gem16_core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_features(gem16_core PUBLIC cxx_std_20)
gem16_set_project_warnings(gem16_core)
gem16_enable_sanitizers(gem16_core)

if(GEM16_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "GEM16_ENABLE_CUDA=ON, but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  gem16_configure_cuda_architectures()
  target_sources(gem16_core PRIVATE
    src/cuda/attention/decode_sm120.cu
    src/cuda/attention/prefill_local_sm120.cu
    src/cuda/attention/prefill_global_sm120.cu
    src/cuda/attention/reference.cu
    src/cuda/dispatch.cu
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
    src/cuda/mtp/verify.cu
    src/cuda/nvfp4/checkpoint_probe.cu
    src/cuda/nvfp4/gemv.cu
    src/cuda/nvfp4/mlp.cu
    src/cuda/nvfp4/mlp_checkpoint_probe.cu
    src/cuda/nvfp4/reference.cu
    src/cuda/nvfp4/sm120.cu
    src/cuda/nvfp4/cutlass_sm120.cu
    src/cuda/output_head.cu
    src/cuda/sampling/sampling.cu
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
    )
  endif()
  target_link_libraries(gem16_core PRIVATE CUDA::cudart CUDA::nvtx3)
  target_compile_definitions(gem16_core
    PUBLIC GEM16_HAS_CUDA=1
    PRIVATE GEM16_COMPILED_CUDA_ARCH="${GEM16_CUDA_ARCHITECTURE}"
  )
  set_target_properties(gem16_core PROPERTIES
    CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
  )
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

include(CTest)
if(BUILD_TESTING AND GEM16_BUILD_TESTS)
  add_executable(gem16-unit-tests
    tests/unit/config_test.cpp
    tests/unit/fp8_test.cpp
    tests/unit/json_test.cpp
    tests/unit/layer_test.cpp
    tests/unit/memory_plan_test.cpp
    tests/unit/mtp_scheduler_test.cpp
    tests/unit/nvfp4_test.cpp
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
  target_link_libraries(gem16-unit-tests PRIVATE gem16_core)
  gem16_set_project_warnings(gem16-unit-tests)
  gem16_enable_sanitizers(gem16-unit-tests)
  add_test(NAME gem16-unit COMMAND gem16-unit-tests)

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
  endif()
endif()
