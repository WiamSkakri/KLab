#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iomanip>

// Error checking macros
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: status %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * @brief Performs matrix multiplication using cuBLAS and measures execution time.
 * 
 * @param m Number of rows in Matrix A and Matrix C.
 * @param n Number of columns in Matrix B and Matrix C.
 * @param k Number of columns in Matrix A and rows in Matrix B.
 * @param iterations Number of multiplication iterations to perform.
 * @return double Average execution time in seconds over the specified iterations.
 */
double perform_gemm(int m, int n, int k, int iterations = 50) {
    // Random number generation setup
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Allocate host memory
    std::vector<float> h_A(m * k, 0.0f);
    std::vector<float> h_B(k * n, 0.0f);
    std::vector<float> h_C(m * n, 0.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));

    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Scalars for GEMM
    float alpha = 1.0f;
    float beta = 0.0f;

    // Total duration accumulator
    std::chrono::duration<double> total_duration = std::chrono::duration<double>::zero();

    // Warm-up iterations
    for (int warm = 0; warm < 2; ++warm) {
        // Initialize matrices A and B with random numbers
        for (auto &val : h_A) val = dist(rng);
        for (auto &val : h_B) val = dist(rng);

        // Copy matrices to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, m * n * sizeof(float)));

        // Perform GEMM
        CUBLAS_CHECK(cublasSgemm(handle, 
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, m, k, 
                                  &alpha, 
                                  d_B, n, 
                                  d_A, k, 
                                  &beta, 
                                  d_C, n));
    }

    // Main timing loop
    for (int iteration = 0; iteration < iterations; ++iteration) {
        // Initialize matrices A and B with random numbers
        for (auto &val : h_A) val = dist(rng);
        for (auto &val : h_B) val = dist(rng);

        // Copy matrices to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, m * n * sizeof(float)));

        // Start timer
        auto start = std::chrono::high_resolution_clock::now();

        // Perform matrix multiplication using cublasSgemm
        CUBLAS_CHECK(cublasSgemm(handle, 
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, m, k, 
                                  &alpha, 
                                  d_B, n, 
                                  d_A, k, 
                                  &beta, 
                                  d_C, n));

        // Synchronize to ensure completion
        CUDA_CHECK(cudaDeviceSynchronize());

        // End timer
        auto end = std::chrono::high_resolution_clock::now();

        // Accumulate duration
        total_duration += (end - start);
    }

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Calculate average execution time
    double average_time = total_duration.count() / iterations;
    return average_time;
}

int main(int argc, char *argv[]) {
    // Ensure correct number of arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " M N K\n";
        return EXIT_FAILURE;
    }

    // Parse and validate input arguments
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);
    if (m <= 0 || n <= 0 || k <= 0) {
        std::cerr << "Error: M, N, and K must be positive integers.\n";
        return EXIT_FAILURE;
    }

    // Perform GEMM and measure average execution time
    double avg_time = perform_gemm(m, n, k);

    // Display the results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Average execution time over 50 iterations: " << avg_time << " seconds\n";

    return EXIT_SUCCESS;
}