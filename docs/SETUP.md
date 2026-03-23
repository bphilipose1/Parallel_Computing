# Setup Guide

## GPU Sorting (`GPU_ACCEL_SORTING/`)

### Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (`nvcc`)
- C++ compiler (gcc/g++)

### Build & Run

```bash
cd GPU_ACCEL_SORTING
nvcc GPU_Sorting.cu -o gpu_sorting
./gpu_sorting
```

**Note:** Input file is currently hardcoded in `main()` as `x_y_16.csv`. Change to your CSV file path before compiling.

**Input format:**
```
x,y
1.5,2.3
3.2,1.8
...
```

**Output:** `x_y_scan.csv` — sorted + prefix-summed data

---

## MPI K-Means Clustering (`MPI_Clustering_Training/`)

### Requirements
- MPI implementation (OpenMPI or MPICH)
- C++ compiler with MPI support (`mpic++`)

### Install MPI

**Ubuntu/Debian:**
```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```

**macOS:**
```bash
brew install open-mpi
```

### Build & Run

```bash
cd MPI_Clustering_Training
make

# Run sequential test
./kmean_color_test

# Run MPI parallel version (4 processes)
mpirun -np 4 ./hw5
```

### Output

- `kmean_colors.html` — Sequential clustering result visualization
- `kmean_colors_MPI.html` — MPI clustering result visualization

---

## Troubleshooting

- **`nvcc: command not found`** → Install CUDA Toolkit and add to PATH
- **`mpicc: command not found`** → Install OpenMPI (see above)
- **`hw6.lock` error** — Remove the lab file lock check in `main()` (line ~200 in `GPU_Sorting.cu`)
