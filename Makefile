.PHONY: all build clean test help install

# Build configuration
BUILD_DIR := build
CMAKE := cmake
MAKE := make
NP := $(shell nproc)

help:
	@echo "Parallel Computing - Build Targets"
	@echo "===================================="
	@echo "make build          - Build both projects (CMake + Make)"
	@echo "make clean          - Clean build artifacts"
	@echo "make test           - Run test suite"
	@echo "make install        - Install binaries to /usr/local/bin"
	@echo "make help           - Show this help"
	@echo ""
	@echo "Project structure:"
	@echo "  GPU_ACCEL_SORTING/    - CUDA-based parallel sorting"
	@echo "  MPI_Clustering_Training/ - MPI k-means clustering"

all: build

build: $(BUILD_DIR)/Makefile
	@echo "Building projects..."
	@cd $(BUILD_DIR) && $(MAKE) -j$(NP)
	@echo "✓ Build successful"
	@echo "Executables:"
	@echo "  GPU:  $(BUILD_DIR)/GPU_ACCEL_SORTING/gpu_sorting"
	@echo "  MPI:  $(BUILD_DIR)/MPI_Clustering_Training/hw5"

$(BUILD_DIR)/Makefile:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) .. -DCMAKE_BUILD_TYPE=Release

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@find . -name "*.o" -delete
	@find . -name "*.out" -delete
	@find . -name "__pycache__" -type d -delete
	@echo "✓ Clean complete"

test: build
	@echo "Running tests..."
	@cd $(BUILD_DIR) && ctest --output-on-failure
	@echo "✓ Tests passed"

install: build
	@echo "Installing binaries..."
	@cd $(BUILD_DIR) && $(CMAKE) --install . --prefix /usr/local
	@echo "✓ Installation complete"
	@echo "Binaries installed to /usr/local/bin"

rebuild: clean build

debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) .. -DCMAKE_BUILD_TYPE=Debug
	@cd $(BUILD_DIR) && $(MAKE) -j$(NP)
	@echo "✓ Debug build complete"

format:
	@echo "Formatting code..."
	@clang-format -i GPU_ACCEL_SORTING/*.cu MPI_Clustering_Training/*.cpp || echo "clang-format not available"

.DEFAULT_GOAL := build
