cmake_minimum_required(VERSION 3.28)

project(gem16gb VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(GEM16GB_ENABLE_CUDA "Build the CUDA capability path" OFF)
option(GEM16GB_BUILD_TESTS "Build host tests" ON)
option(GEM16GB_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(GEM16GB_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)

include(CompilerWarnings)
include(Sanitizers)
include(CudaArchitectures)

add_library(gem16gb_core STATIC
  src/numeric/fp8.cpp
  src/numeric/layer.cpp
  src/numeric/nvfp4.cpp
  src/cuda/nvfp4/sm120_layout.cpp
  src/model/checkpoint_loader.cpp
  src/model/config.cpp
  src/model/manifest.cpp
  src/model/safetensors.cpp
  src/model/tokenizer.cpp
  src/runtime/capabilities.cpp
  src/runtime/memory.cpp
  src/runtime/memory_plan.cpp
  src/util/json.cpp
)
if(WIN32)
  target_sources(gem16gb_core PRIVATE src/platform/mapped_file_windows.cpp)
else()
  target_sources(gem16gb_core PRIVATE src/platform/mapped_file_posix.cpp)
endif()
# GCC 16 emits a false positive from std::variant's move constructor at -O3.
# Keep the suppression local to the JSON translation unit.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_property(SOURCE src/util/json.cpp APPEND PROPERTY COMPILE_OPTIONS -Wno-maybe-uninitialized)
endif()
target_include_directories(gem16gb_core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_features(gem16gb_core PUBLIC cxx_std_20)
gem16gb_set_project_warnings(gem16gb_core)
gem16gb_enable_sanitizers(gem16gb_core)

if(GEM16GB_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "GEM16GB_ENABLE_CUDA=ON, but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  gem16gb_configure_cuda_architectures()
  target_sources(gem16gb_core PRIVATE
    src/cuda/attention/sm120.cu
    src/cuda/dispatch.cu
    src/cuda/inference.cu
    src/cuda/fp8/checkpoint_probe.cu
    src/cuda/fp8/reference.cu
    src/cuda/fp8/sm120.cu
    src/cuda/layer/reference.cu
    src/cuda/layer/checkpoint_probe.cu
    src/cuda/nvfp4/checkpoint_probe.cu
    src/cuda/nvfp4/gemv.cu
    src/cuda/nvfp4/mlp.cu
    src/cuda/nvfp4/mlp_checkpoint_probe.cu
    src/cuda/nvfp4/reference.cu
    src/cuda/nvfp4/sm120.cu
  )
  if(MSVC)
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
    set_property(SOURCE src/cuda/attention/sm120.cu APPEND PROPERTY COMPILE_OPTIONS
      -Xcompiler=/wd4211
    )
    set_property(SOURCE src/cuda/layer/reference.cu APPEND PROPERTY COMPILE_OPTIONS
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
  endif()
  target_link_libraries(gem16gb_core PRIVATE CUDA::cudart CUDA::nvtx3)
  target_compile_definitions(gem16gb_core
    PUBLIC GEM16GB_HAS_CUDA=1
    PRIVATE GEM16GB_COMPILED_CUDA_ARCH="${GEM16GB_CUDA_ARCHITECTURE}"
  )
  set_target_properties(gem16gb_core PROPERTIES
    CUDA_ARCHITECTURES "${GEM16GB_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
  )
  message(STATUS "gem16gb CUDA capability scaffold: enabled for ${CMAKE_CUDA_ARCHITECTURES}")
else()
  target_sources(gem16gb_core PRIVATE src/runtime/inference_stub.cpp)
  target_compile_definitions(gem16gb_core PUBLIC GEM16GB_HAS_CUDA=0)
  message(STATUS "gem16gb CUDA capability scaffold: disabled")
endif()

foreach(tool IN ITEMS inspect run bench chat)
  add_executable(gem16gb-${tool} src/cli/${tool}_main.cpp)
  target_link_libraries(gem16gb-${tool} PRIVATE gem16gb_core)
  gem16gb_set_project_warnings(gem16gb-${tool})
  gem16gb_enable_sanitizers(gem16gb-${tool})
endforeach()
target_include_directories(gem16gb-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

include(CTest)
if(BUILD_TESTING AND GEM16GB_BUILD_TESTS)
  add_executable(gem16gb-unit-tests
    tests/unit/fp8_test.cpp
    tests/unit/json_test.cpp
    tests/unit/layer_test.cpp
    tests/unit/memory_plan_test.cpp
    tests/unit/nvfp4_test.cpp
    tests/unit/safetensors_test.cpp
    tests/unit/sm120_layout_test.cpp
    tests/unit/tokenizer_test.cpp
    tests/unit/test_main.cpp
  )
  if(WIN32)
    target_sources(gem16gb-unit-tests PRIVATE tests/unit/windows_utf8_test.cpp)
  endif()
  target_include_directories(gem16gb-unit-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit")
  target_link_libraries(gem16gb-unit-tests PRIVATE gem16gb_core)
  gem16gb_set_project_warnings(gem16gb-unit-tests)
  gem16gb_enable_sanitizers(gem16gb-unit-tests)
  add_test(NAME gem16gb-unit COMMAND gem16gb-unit-tests)

  if(GEM16GB_ENABLE_CUDA)
    add_executable(gem16gb-cuda-tests tests/cuda/nvfp4_reference_test.cu)
    target_include_directories(gem16gb-cuda-tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(gem16gb-cuda-tests PRIVATE gem16gb_core CUDA::cudart)
    set_target_properties(gem16gb-cuda-tests PROPERTIES
      CUDA_ARCHITECTURES "${GEM16GB_CUDA_ARCHITECTURE}"
    )
    gem16gb_set_project_warnings(gem16gb-cuda-tests)
    add_test(NAME gem16gb-cuda COMMAND gem16gb-cuda-tests)
  endif()
endif()
