# Project Status

## Projects

### GPU_ACCEL_SORTING
- Implements parallel scan + bitonic sort using CUDA
- Sorts 2D point data from CSV by x value, then computes prefix sum on y values
- Status: Complete (coursework assignment)

### MPI_Clustering_Training
- Implements K-Means color clustering using MPI
- Sequential (`KMeans.h`) and parallel (`KMeansMPI.h`) versions
- Visualizes results as HTML
- Status: Complete (coursework assignment)

## What Was Added (Cleanup)
- `.gitignore` — Excludes build artifacts, executables, output files
- `README.md` — Project overview and quick start
- `docs/` — Algorithm explanations, setup guide
- `CMakeLists.txt` files — CMake build support
- `Makefile` — Unified build commands
- `.github/workflows/` — CI/CD pipeline (GitHub Actions)
- `GPU_ACCEL_SORTING/sample_data.csv` — Sample input data
