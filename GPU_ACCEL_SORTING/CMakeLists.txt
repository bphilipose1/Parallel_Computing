# GPU Acceleration Sorting

# Add CUDA executable
add_executable(gpu_sorting GPU_Sorting.cu)

# Link CUDA libraries
target_link_libraries(gpu_sorting ${CUDA_LIBRARIES})

# CUDA compile options
set_target_properties(gpu_sorting PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "35;50;60;70;80"
)

# Installation
install(TARGETS gpu_sorting DESTINATION bin)

# Add test (optional)
enable_testing()
add_test(NAME gpu_sorting_basic COMMAND gpu_sorting)
