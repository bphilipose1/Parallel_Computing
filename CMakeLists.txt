cmake_minimum_required(VERSION 3.10)
project(ParallelComputing CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find optional CUDA
find_package(CUDA)
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(MPI REQUIRED)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")
endif()

# CUDA specific
if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_35")
    add_subdirectory(GPU_ACCEL_SORTING)
endif()

# Subdirectories
add_subdirectory(MPI_Clustering_Training)

# Info message
message(STATUS "Parallel Computing Build Configuration:")
if(CUDA_FOUND)
    message(STATUS "  CUDA: found (${CUDA_VERSION})")
else()
    message(STATUS "  CUDA: not found - GPU acceleration disabled")
endif()
message(STATUS "  MPI: ${MPI_FOUND}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
