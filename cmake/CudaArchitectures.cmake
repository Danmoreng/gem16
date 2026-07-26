set(GEM16_CUDA_ARCHITECTURE "120a" CACHE STRING "Single CUDA architecture for performance builds")

function(gem16_configure_cuda_architectures)
  if(NOT GEM16_CUDA_ARCHITECTURE MATCHES "^120a?$")
    message(FATAL_ERROR "The phase-one CUDA build supports only SM120/SM120a, got '${GEM16_CUDA_ARCHITECTURE}'")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}" CACHE STRING "CUDA architectures" FORCE)
  set(CMAKE_CUDA_ARCHITECTURES "${GEM16_CUDA_ARCHITECTURE}" PARENT_SCOPE)
endfunction()
