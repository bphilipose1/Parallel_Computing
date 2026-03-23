# Algorithm Documentation

## GPU-Accelerated Sorting (`GPU_ACCEL_SORTING/`)

### Overview
Uses CUDA to perform a parallel **prefix scan** followed by a **bitonic sort** on 2D point data (x, y coordinates) loaded from a CSV file.

### Step-by-Step

1. **Load data** from CSV into an array of `X_Y` structs
2. **Pad to power of 2** — required by bitonic sort (pads with lowest float sentinel values)
3. **Copy to GPU** (`cudaMallocManaged`)
4. **Bitonic sort** by x value on GPU
5. **Prefix scan** on the sorted y values
6. **Write results** back to output CSV

### Bitonic Sort

Parallel comparison-based sort. Works by creating a "bitonic sequence" (first ascending, then descending) and repeatedly merging/sorting it.

- Requires input size to be a power of 2 (hence the padding)
- Sorting is done by `x` value
- GPU kernel: `bitonic()`

**Reference:** https://en.wikipedia.org/wiki/Bitonic_sort

### Prefix Scan (Parallel Prefix Sum)

Computes a running cumulative sum of the `scan` field (initialized to `y` values) across the sorted array.

- Uses a two-tier approach: scan within each block, then adjust across blocks
- GPU kernels: `scanCUDA()` → `adjust_subarrays()`

**Reference:** https://en.wikipedia.org/wiki/Prefix_sum#Parallel_algorithm

### Data Structure

```cpp
struct X_Y {
    int n;       // original index
    float x, y; // coordinates
    float scan;  // scan value (initialized to y)
};
```

### Key Constants

```cpp
const int MAX_BLOCK_SIZE = 1024;  // Threads per block
const int MAX_BLOCK = 1024;       // Max grid blocks
```

---

## MPI K-Means Clustering (`MPI_Clustering_Training/`)

### Overview
Implements K-Means clustering for color data using both a sequential version (`KMeans.h`) and a distributed MPI version (`KMeansMPI.h`). Designed to cluster colors (RGB, treated as 3D points) from image data.

### K-Means Algorithm

1. **Initialize** — Randomly pick `k` centroids from the data
2. **Assign** — Each data point assigned to nearest centroid (Euclidean distance)
3. **Update** — Recalculate each centroid as the mean of its assigned points
4. **Repeat** — Up to `MAX_FIT_STEPS = 300` iterations or until convergence

**Reference:** https://en.wikipedia.org/wiki/K-means_clustering

### Sequential Version (`KMeans.h`)

Runs on a single process. Standard Lloyd's algorithm implementation.

### MPI Parallel Version (`KMeansMPI.h`)

Distributes data across MPI processes:

1. **Rank 0** initializes centroids, broadcasts to all processes
2. **All ranks** compute local assignments and partial centroid sums
3. **MPI_Allreduce** aggregates partial sums across all processes
4. **All ranks** update centroids, check convergence
5. **Repeat** until done

**MPI functions used:**
- `MPI_Comm_rank` / `MPI_Comm_size` — Process identity
- `MPI_Bcast` — Broadcast centroids from rank 0
- `MPI_Allreduce` — Sum partial cluster counts/sums across all ranks

**Reference:** https://en.wikipedia.org/wiki/Message_Passing_Interface

### Color Data

Colors are stored as `std::array<u_char, d>` where `d = 3` (RGB channels). Distance between two colors uses Euclidean distance in RGB space.

### Template Parameters

```cpp
template <int k, int d>
class KMeansMPI { ... };
// k = number of clusters
// d = dimensions (3 for RGB)
```

### Output

Results are visualized as HTML files (`kmean_colors.html`, `kmean_colors_MPI.html`) showing the dominant color clusters found.
