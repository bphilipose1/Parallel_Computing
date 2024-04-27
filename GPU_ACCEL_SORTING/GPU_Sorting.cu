/**
 * Program uses CUDA to handle Scanning and Sorting of a large amount of elements efficiently
 */

#include <iostream>
#include <fstream>
#include <sys/file.h> 
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <limits>
#include <iomanip>
#include <cmath>
using namespace std;

const int MAX_BLOCK_SIZE = 1024;
const int MAX_BLOCK = 1024;

/**
 * Struct to hold X and Y value along with an additional scan value, and pairs original order/rank
 */
struct X_Y {
    int n;
    float x, y;
    float scan;
};


/**
 * Loads data from a CSV file into an array of X_Y objects
 * 
 * @param filename Name of the file
 * @param n Number of Elements read into structure
 * @return Array of X_Y objects
 */
X_Y* load_data(const string& filename, int& n) {
    ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        n = 0;
        return nullptr;
    }

    vector<float> xTempVect, yTempVect;
    string line;

    getline(file, line); // Skip first line
    while (getline(file, line)) {
        istringstream iss(line);
        string xTemp, yTemp;

        if (getline(iss, xTemp, ',') && getline(iss, yTemp)) {
            xTempVect.push_back(stof(xTemp));
            yTempVect.push_back(stof(yTemp));
        }
    }
    file.close();

    n = xTempVect.size(); // Number of elements read
    X_Y* data = new X_Y[n]; 

    // Copy data into X_Y array
    for (int i = 0; i < n; ++i) {
        data[i].x = xTempVect[i];
        data[i].y = yTempVect[i];
        data[i].n = i+1;
        data[i].scan = yTempVect[i];
    }
    return data;
}

/**
 * Pads data to the next power of 2
 * 
 * @param originalData The original array
 * @param originalCount Number of elements in original  array
 * @param paddedCount New total number of elements after padding
 * @return Array of X_Y objects with padded data
 */
X_Y* padDataToPowerOf2(const X_Y* originalData, int originalCount, int& paddedCount) {
    int nearestPowerOf2 = pow(2, ceil(log2(originalCount)));
    paddedCount = nearestPowerOf2;

    X_Y* paddedData = new X_Y[nearestPowerOf2];

    for (int i = 0; i < originalCount; ++i) {
        paddedData[i] = originalData[i];
    }

    // Pad remaining elements with min float values (Due to data seeming to not have any negative values, lowest float val seemed like good flag)
    X_Y minXYValue = {-1, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
    for (int i = originalCount; i < nearestPowerOf2; ++i) {
        paddedData[i] = minXYValue;
    }
    return paddedData;
}

/**
 * Writes the finalized processed data into output CSV file
 * 
 * @param outfile Output file name
 * @param data The array of X_Y objects to be written to the file
 * @param n The number of elements to write
 */
void dump_data(const string& outfile, X_Y* data, int n) {
    std::ofstream ofile(outfile);
    if (!ofile) {
        return;
    }
    
    ofile << "n,x,y,scan\n";
    for (int i = 0; i < n; ++i) {
        // Set precision of up to 6 decimal points when writing back
        ofile << std::fixed << std::setprecision(6);
        ofile << data[i].n << "," << data[i].x << "," << data[i].y << "," << data[i].scan << '\n';
    }
}

/**
 * CUDA device function to swap two elements in an array of X_Y objects
 * 
 * @param data The array of X_Y object
 * @param a First element index
 * @param b Second element index
 */
__device__ void swap(X_Y *data, int a, int b) {
    X_Y temp;

    // Manually swap each field
    temp.n = data[a].n;
    data[a].n = data[b].n;
    data[b].n = temp.n;

    temp.x = data[a].x;
    data[a].x = data[b].x;
    data[b].x = temp.x;

    temp.y = data[a].y;
    data[a].y = data[b].y;
    data[b].y = temp.y;

    temp.scan = data[a].scan;
    data[a].scan = data[b].scan;
    data[b].scan = temp.scan;
}

/**
 * GPU bitonic sort
 * 
 * @param data Data array of X_Y objects
 * @param k The stage of the bitonic sort
 * @param n The number of elements in the data array
 * @param startJ The starting value of j for the current stage
 * @param endJ The ending value of j for the current stage
 */
__global__ void bitonic(X_Y *data, int k, int n, int startJ, int endJ) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    for (int j = startJ; j > endJ; j /= 2) {
        int ixj = i ^ j;
        if (ixj > i) {
            if ((i & k) == 0 && data[i].x > data[ixj].x)
                swap(data, i, ixj);
            if ((i & k) != 0 && data[i].x < data[ixj].x)
                swap(data, i, ixj);
        }
        __syncthreads();
    }
}


/**
 * Sorts an array of X_Y pairs by their x values using the bitonic sort algorithm on the CPU/GPU
 * 
 * @param deviceData Array of X_Y objects
 * @param n Number of elements in the array
 */
void sort(X_Y *deviceData, int n) {
    int numBlocks = (n + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    numBlocks = min(numBlocks, MAX_BLOCK);
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            if (j > MAX_BLOCK_SIZE/2) { // J has multi-block reach
                bitonic<<<numBlocks, MAX_BLOCK_SIZE>>>(deviceData, k, n, j, j-1);
                cudaDeviceSynchronize();
            } else { // J is within single-block reach
                bitonic<<<numBlocks, MAX_BLOCK_SIZE>>>(deviceData, k, n, j, 0);
                cudaDeviceSynchronize();
                break; 
            }
        }
        cudaDeviceSynchronize();
    }
}

/**
 * GPU performs a scan operation on an array of X_Y objects
 * 
 * @param data Array of X_Y objects
 * @param blockResults Array to store the results of the scan for each block
 * @param n The number of elements in the data array
 * @param prefix True to store prefix scan of each block, false otherwise
 */
__global__ void scanCUDA(X_Y *data, X_Y *blockResults, int n, bool prefix) {
    __shared__ float local[MAX_BLOCK_SIZE];
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    int index = threadIdx.x;

    local[index] = data[gindex].scan;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float addend = 0;
        if (index >= stride)
            addend = local[index - stride];
        __syncthreads();
        local[index] += addend;
    }
    data[gindex].scan = local[index];

    
    if ((index == blockDim.x - 1) && prefix) { // Write the last element of the block's scan result to blockResults
        blockResults[blockIdx.x].scan = local[index];
    }
}

/**
 * GPU adjust subarrays after a scan operation
 * 
 * @param data Array of X_Y objects
 * @param blockResults Array holding scan results of each block
 * @param n The number of elements in the data array
 */
__global__ void adjust_subarrays(X_Y *data, X_Y *blockResults, int n) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0) {
        data[globalIdx].scan += blockResults[blockIdx.x-1].scan;
    }
}

/**
 * Performs a scan operation on an array of X_Y objects using the GPU
 * 
 * @param data Array of X_Y objects
 * @param n The number of elements in the array
 */
void scan(X_Y *data, int n) {
    int numBlocks = (n + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    
    X_Y *blockResults;
    cudaMallocManaged(&blockResults, numBlocks * sizeof(X_Y));

    // First-tier scan
    scanCUDA<<<numBlocks, MAX_BLOCK_SIZE>>>(data, blockResults, n, true);
    cudaDeviceSynchronize();

    // Second-tier scan (Optional)
    if (numBlocks > 1) {
        scanCUDA<<<1, numBlocks>>>(blockResults, nullptr, numBlocks, false);
        cudaDeviceSynchronize();
    }

    // Adjust the original sub-arrays with the prefix sums of the block results
    adjust_subarrays<<<MAX_BLOCK, MAX_BLOCK_SIZE>>>(data, blockResults, n);
    cudaDeviceSynchronize();
    cudaFree(blockResults);
}

/**
 * Main function to execute the CUDA sorting and scanning operations on input dataset
 */
int main()  {
    if (flock(open("/home/fac/lundeenk/hw6.lock", O_RDONLY), LOCK_EX|LOCK_NB) == -1) {
            cout << "someone has hw6 locked--could it be you?" << endl;
            return EXIT_FAILURE;
    }
    
    string infile = "x_y_16.csv";
    //string infile = "x_y_100.csv";
    //string infile = "x_y_1024.csv";
    //string infile = "x_y_10000.csv";
    //string infile = "/home/fac/lundeenk/x_y.csv";

    string outfile = "./x_y_scan.csv";
    
    int n = 0;
    X_Y *data = load_data(infile, n);
    if(data == nullptr) {
        cout << "could not open " << infile << endl;
        return 1;
    }

    int paddedCount;
    X_Y* paddedData = padDataToPowerOf2(data, n, paddedCount);
    X_Y* deviceData;
    size_t size = paddedCount * sizeof(X_Y);
    cudaMallocManaged(&deviceData, size);
    memcpy(deviceData, paddedData, size);

    sort(deviceData, paddedCount);

    // Prepare a new array for scanning that excludes the padded elements for scan and result step
    int startIndex = paddedCount - n;
    X_Y* meaningfulData;
    cudaMallocManaged(&meaningfulData, n * sizeof(X_Y)); 
    memcpy(meaningfulData, &deviceData[startIndex], n * sizeof(X_Y)); 

    scan(meaningfulData, n);

    dump_data(outfile, meaningfulData, n);
    
    cudaFree(deviceData);
    cudaFree(meaningfulData);
    delete[] data;
    delete[] paddedData;
    return 0;
}
