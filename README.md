# Parallel Computing

High-performance parallel computing implementations using CUDA and MPI.

## Projects

### 1. GPU-Accelerated Sorting
Efficient parallel sorting and scanning using CUDA on GPU hardware.

**Features:**
- Parallel scan algorithm (prefix sum)
- GPU-accelerated sorting
- CSV data I/O
- Configurable block sizes (up to 1024 threads)

**Tech Stack:** CUDA, C++

**Performance:** Designed for large datasets (10K+ elements)

### 2. MPI K-Means Clustering
Distributed k-means clustering implementation with MPI parallelization.

**Features:**
- Sequential and MPI-parallel implementations
- Color-based clustering
- Configurable cluster count
- HTML output visualization

**Tech Stack:** MPI, C++

**Performance:** Scales with processor count

## Quick Start

### Prerequisites

**GPU Sorting:**
- NVIDIA GPU with CUDA Compute Capability ≥ 3.0
- CUDA Toolkit 10.0+ (ideally 11.0+)
- gcc/g++ (C++11 or higher)

**MPI Clustering:**
- MPI implementation (OpenMPI or MPICH)
- C++11 compliant compiler
- Standard POSIX system

### Build

```bash
# GPU Sorting
cd GPU_ACCEL_SORTING
make

# MPI Clustering
cd ../MPI_Clustering_Training
make

# Run tests
./kmean_color_test
```

### Example Usage

**GPU Sorting:**
```bash
./GPU_ACCEL_SORTING/gpu_sorting data.csv output.csv
```

**MPI K-Means:**
```bash
mpirun -np 4 ./MPI_Clustering_Training/hw5
```

## Project Structure

```
.
├── GPU_ACCEL_SORTING/          # GPU-accelerated sorting
│   └── GPU_Sorting.cu
├── MPI_Clustering_Training/     # MPI k-means clustering
│   ├── hw5.cpp
│   ├── Color.cpp
│   ├── KMeans.h
│   ├── KMeansMPI.h
│   ├── Makefile
│   └── test outputs (*.html, *.o)
├── docs/                        # Documentation (TODO)
├── README.md                    # This file
└── .gitignore
```

## Documentation

- [Algorithm Details](docs/ALGORITHM.md) - Coming soon
- [Performance Analysis](docs/PERFORMANCE.md) - Coming soon
- [Setup Guide](docs/SETUP.md) - Coming soon

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Status

🚧 **Completed** — Courses have been completed.

## Author

Benjamin Philipose
