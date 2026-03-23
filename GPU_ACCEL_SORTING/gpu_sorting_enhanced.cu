/**
 * GPU-Accelerated Sorting using CUDA
 * Implements parallel scan (prefix sum) and bitonic sort
 * 
 * Usage: ./gpu_sorting <input.csv> <output.csv>
 * Input format: x,y (with header)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <limits>
#include <iomanip>
#include <cstring>

using namespace std;

// ============================================================================
// Configuration Constants
// ============================================================================

const int MAX_BLOCK_SIZE = 1024;        // Threads per block (GPU limit)
const int MAX_GRID_SIZE = 1024;         // Max blocks in grid
const float EPSILON = 1e-6f;            // Floating point comparison tolerance

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Represents a 2D point with scan value
 * n: original index
 * x, y: coordinates
 * scan: scan/sorted value
 */
struct X_Y {
    int n;
    float x, y;
    float scan;
};

// ============================================================================
// Error Handling Macros
// ============================================================================

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ============================================================================
// File I/O Functions
// ============================================================================

/**
 * Load data from CSV file
 * Format: x,y (with header line)
 * 
 * @param filename Input CSV filename
 * @param n [out] Number of elements loaded
 * @return Allocated X_Y array (caller must free)
 */
X_Y* load_data(const string& filename, int& n) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "ERROR: Could not open file: " << filename << endl;
        n = 0;
        return nullptr;
    }

    vector<float> xVec, yVec;
    string line;

    // Skip header
    if (!getline(file, line)) {
        cerr << "ERROR: Empty file" << endl;
        return nullptr;
    }

    // Parse CSV
    while (getline(file, line)) {
        if (line.empty()) continue;
        
        istringstream iss(line);
        string xStr, yStr;

        if (getline(iss, xStr, ',') && getline(iss, yStr)) {
            try {
                xVec.push_back(stof(xStr));
                yVec.push_back(stof(yStr));
            } catch (const exception& e) {
                cerr << "WARNING: Skipping malformed line: " << line << endl;
                continue;
            }
        }
    }
    file.close();

    n = xVec.size();
    if (n == 0) {
        cerr << "ERROR: No valid data in file" << endl;
        return nullptr;
    }

    X_Y* data = new X_Y[n];
    for (int i = 0; i < n; ++i) {
        data[i].n = i + 1;
        data[i].x = xVec[i];
        data[i].y = yVec[i];
        data[i].scan = yVec[i];
    }

    cout << "Loaded " << n << " points from " << filename << endl;
    return data;
}

/**
 * Pad data to next power of 2 (required for scan algorithm)
 * 
 * @param original Original data array
 * @param originalCount Original element count
 * @param paddedCount [out] Padded element count
 * @return New padded array (caller must free)
 */
X_Y* padDataToPowerOf2(const X_Y* original, int originalCount, int& paddedCount) {
    int powerOf2 = 1;
    while (powerOf2 < originalCount) {
        powerOf2 *= 2;
    }
    paddedCount = powerOf2;

    X_Y* padded = new X_Y[paddedCount];
    
    // Copy original data
    memcpy(padded, original, originalCount * sizeof(X_Y));
    
    // Pad with sentinel values (lowest float for comparison)
    X_Y sentinel = {-1, numeric_limits<float>::lowest(), 
                    numeric_limits<float>::lowest(), 
                    numeric_limits<float>::lowest()};
    
    for (int i = originalCount; i < paddedCount; ++i) {
        padded[i] = sentinel;
    }

    cout << "Padded " << originalCount << " elements to " << paddedCount << endl;
    return padded;
}

/**
 * Write sorted data to CSV file
 * 
 * @param filename Output filename
 * @param data Data array
 * @param count Number of elements (excluding padding)
 */
void write_data(const string& filename, const X_Y* data, int count) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "ERROR: Could not open output file: " << filename << endl;
        return;
    }

    file << "n,x,y,scan_value\n";
    for (int i = 0; i < count; ++i) {
        file << data[i].n << ","
             << fixed << setprecision(6) << data[i].x << ","
             << data[i].y << ","
             << data[i].scan << "\n";
    }
    file.close();

    cout << "Wrote " << count << " elements to " << filename << endl;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Up-sweep phase of scan (reduction)
 * Block-wise parallel scan
 */
__global__ void kernel_up_sweep(X_Y* data, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    int compare_idx = idx + offset;
    
    if (compare_idx < n) {
        X_Y a = data[idx];
        X_Y b = data[compare_idx];
        
        // Sum scan values
        X_Y result = a;
        result.scan = a.scan + b.scan;
        
        // Ensure proper sync
        __syncthreads();
        
        data[compare_idx] = result;
    }
}

/**
 * Down-sweep phase of scan (distribution)
 */
__global__ void kernel_down_sweep(X_Y* data, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n || idx + offset >= n) return;
    
    X_Y a = data[idx];
    X_Y b = data[idx + offset];
    
    X_Y result = b;
    result.scan = a.scan + b.scan;
    
    __syncthreads();
    
    data[idx + offset] = result;
}

/**
 * Bitonic sort kernel (simplified for X_Y structs)
 */
__global__ void kernel_bitonic_sort(X_Y* data, int n, int p, int q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Determine comparison direction
    int d = (idx >> p) & 1;
    int comp_idx = idx ^ (1 << q);

    if (comp_idx > idx && comp_idx < n) {
        X_Y a = data[idx];
        X_Y b = data[comp_idx];

        bool swap = (d == 0) ? (a.scan > b.scan) : (a.scan < b.scan);

        if (swap) {
            X_Y temp = a;
            data[idx] = b;
            data[comp_idx] = temp;
        }
    }
}

// ============================================================================
// Host Functions
// ============================================================================

/**
 * Parallel scan on GPU using Blelloch scan algorithm
 * 
 * @param d_data GPU memory pointer
 * @param n Number of elements
 */
void gpu_scan(X_Y* d_data, int n) {
    cout << "Starting GPU scan..." << endl;
    
    // Up-sweep (reduction)
    for (int offset = 1; offset < n; offset *= 2) {
        int blocks = (n + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
        kernel_up_sweep<<<blocks, MAX_BLOCK_SIZE>>>(d_data, n, offset);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    // Down-sweep
    for (int offset = n / 2; offset >= 1; offset /= 2) {
        int blocks = (n + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
        kernel_down_sweep<<<blocks, MAX_BLOCK_SIZE>>>(d_data, n, offset);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cout << "GPU scan complete" << endl;
}

/**
 * Parallel sort on GPU using bitonic sort
 * 
 * @param d_data GPU memory pointer
 * @param n Number of elements (must be power of 2)
 */
void gpu_sort(X_Y* d_data, int n) {
    cout << "Starting GPU bitonic sort..." << endl;
    
    // Bitonic sort phases
    for (int p = 1; p < n; p *= 2) {
        for (int q = p; q >= 1; q /= 2) {
            int blocks = (n + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
            kernel_bitonic_sort<<<blocks, MAX_BLOCK_SIZE>>>(d_data, n, p, q);
            CUDA_CHECK(cudaPeekAtLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cout << "GPU sort complete" << endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input.csv> <output.csv>" << endl;
        cerr << "Input format: CSV with header 'x,y'" << endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    // Load data
    int n = 0;
    X_Y* h_data = load_data(input_file, n);
    
    if (!h_data || n == 0) {
        cerr << "ERROR: Failed to load data" << endl;
        return 1;
    }

    // Pad to power of 2
    int n_padded = 0;
    X_Y* h_data_padded = padDataToPowerOf2(h_data, n, n_padded);

    // Allocate GPU memory
    X_Y* d_data = nullptr;
    size_t bytes = n_padded * sizeof(X_Y);
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_data_padded, bytes, cudaMemcpyHostToDevice));
    cout << "Transferred " << bytes / (1024.0 * 1024.0) << " MB to GPU" << endl;

    // Process on GPU
    gpu_scan(d_data, n_padded);
    gpu_sort(d_data, n_padded);

    // Copy back to host
    CUDA_CHECK(cudaMemcpy(h_data_padded, d_data, bytes, cudaMemcpyDeviceToHost));
    cout << "Transferred " << bytes / (1024.0 * 1024.0) << " MB from GPU" << endl;

    // Write results (only non-padded elements)
    write_data(output_file, h_data_padded, n);

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;
    delete[] h_data_padded;

    cout << "Success!" << endl;
    return 0;
}
