#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;
using namespace std::chrono;
// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
             << ": " << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)
// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        cerr << "cuBLAS error in " << __FILE__ << " at line " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)
int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " n k\n";
        return 1;
    }
    // Parse command line arguments
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    if (n < 1 || k < 1) {
        cout << "Error: Matrix dimensions must be positive\n";
        return 1;
    }
    // Allocate host memory
    vector<cuDoubleComplex> h_A(n * k);
    vector<cuDoubleComplex> h_C(n * n, make_cuDoubleComplex(0.0, 0.0));
    // Initialize matrix A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            h_A[i + j * n] = make_cuDoubleComplex(
                (double)(i + 1) / (j + 1),
                (double)(i + j) / n
            );
        }
    }
    // Allocate device memory
    cuDoubleComplex *d_A, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, n * k * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_C, n * n * sizeof(cuDoubleComplex)));
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Copy input matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), n * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), n * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    // Parameters for zherk
    const int NUM_ITERATIONS = 50;
    vector<double> execution_times;
    execution_times.reserve(NUM_ITERATIONS);
    // Warm-up run
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    CUBLAS_CHECK(cublasZherk(
        handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        n, k, 
        reinterpret_cast<const double*>(&alpha),
        reinterpret_cast<const cuDoubleComplex*>(d_A), n,
        reinterpret_cast<const double*>(&beta),
        reinterpret_cast<cuDoubleComplex*>(d_C), n
    ));
    // Performance measurement iterations
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Reset device memory
        CUDA_CHECK(cudaMemset(d_C, 0, n * n * sizeof(cuDoubleComplex)));
        // Start timer
        auto start = high_resolution_clock::now();
        // Perform Hermitian rank-k update
        CUBLAS_CHECK(cublasZherk(
            handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
            n, k, 
            reinterpret_cast<const double*>(&alpha),
            reinterpret_cast<const cuDoubleComplex*>(d_A), n,
            reinterpret_cast<const double*>(&beta),
            reinterpret_cast<cuDoubleComplex*>(d_C), n
        ));
        // Stop timer
        auto end = high_resolution_clock::now();
        execution_times.push_back(duration_cast<duration<double>>(end - start).count());
    }
    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
    // Output average execution time in seconds
    cout << fixed << accumulate(execution_times.begin(), execution_times.end(), 0.0) / NUM_ITERATIONS << endl;
    return 0;
}