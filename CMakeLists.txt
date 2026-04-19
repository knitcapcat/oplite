cmake_minimum_required(VERSION 3.18)
project(oplite LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to A100 (sm_80). Override with:
#   cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..   # for RTX 40-series / L40
#   cmake -DCMAKE_CUDA_ARCHITECTURES=90 ..   # for H100
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Useful debug info even in Release: line tables for ncu source-correlation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------
# Header-only kernel library
# ------------------------------------------------------------------
add_library(oplite INTERFACE)
target_include_directories(oplite INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(oplite INTERFACE CUDA::cudart)

# ------------------------------------------------------------------
# Tests (correctness vs cuBLAS) and benchmarks (perf vs cuBLAS)
# ------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
add_subdirectory(bench)
