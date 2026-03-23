# MPI K-Means Clustering

# Main executable
add_executable(hw5 hw5.cpp Color.cpp)
target_link_libraries(hw5 MPI::MPI_CXX)
target_include_directories(hw5 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Test executable
add_executable(kmean_color_test kmean_color_test.cpp Color.cpp)
target_link_libraries(kmean_color_test MPI::MPI_CXX)
target_include_directories(kmean_color_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Compiler flags for MPI
set_target_properties(hw5 kmean_color_test PROPERTIES
    COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
    LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
)

# Installation
install(TARGETS hw5 kmean_color_test DESTINATION bin)

# Add tests
enable_testing()
add_test(NAME kmeans_test COMMAND kmean_color_test)
