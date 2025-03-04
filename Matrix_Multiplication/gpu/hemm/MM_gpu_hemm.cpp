#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <complex>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <cstdlib>
#include <ctime>

// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
} while(0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        throw std::runtime_error("cuBLAS error"); \
    } \
} while(0)

void initialize_hermitian_matrix(std::complex<double>* A, int m) {
    srand(time(NULL));
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            if (i == j) {
                A[j * m + i] = std::complex<double>((double)rand()/RAND_MAX, 0.0);
            } else {
                double real = (double)rand()/RAND_MAX;
                double imag = (double)rand()/RAND_MAX;
                A[j * m + i] = std::complex<double>(real, imag);
                A[i * m + j] = std::conj(A[j * m + i]);
            }
        }
    }
}

void initialize_matrix(std::complex<double>* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        double real = (double)rand()/RAND_MAX;
        double imag = (double)rand()/RAND_MAX;
        matrix[i] = std::complex<double>(real, imag);
    }
}

double run_single_iteration(std::complex<double>* A, std::complex<double>* B,
                           std::complex<double>* C, int m, int n) {
    // Device pointers
    std::complex<double> *d_A, *d_B, *d_C;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, m * m * sizeof(std::complex<double>)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, m * n * sizeof(std::complex<double>)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, m * n * sizeof(std::complex<double>)));
    
    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, A, m * m * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, m * n * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, m * n * sizeof(std::complex<double>)));
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Prepare parameters for zhemm
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    
    // Timing using CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    // Perform HEMM
    CUBLAS_CHECK(cublasZhemm(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        m,
        n,
        &alpha,
        reinterpret_cast<cuDoubleComplex*>(d_A),
        m,
        reinterpret_cast<cuDoubleComplex*>(d_B),
        m,
        &beta,
        reinterpret_cast<cuDoubleComplex*>(d_C),
        m
    ));
    
    // Stop timer
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return milliseconds;
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 3) return 1;
        
        const int m = std::atoi(argv[1]);
        const int n = std::atoi(argv[2]);
        const int NUM_ITERATIONS = 50;
        
        if (m <= 0 || n <= 0) return 1;
        
        std::vector<std::complex<double>> A(m * m);
        std::vector<std::complex<double>> B(m * n);
        std::vector<std::complex<double>> C(m * n);
        std::vector<double> execution_times(NUM_ITERATIONS);
        
        initialize_hermitian_matrix(A.data(), m);
        initialize_matrix(B.data(), m, n);
        
        // Warmup iteration
        run_single_iteration(A.data(), B.data(), C.data(), m, n);
        
        // Run iterations
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            std::fill(C.begin(), C.end(), std::complex<double>(0.0, 0.0));
            execution_times[i] = run_single_iteration(A.data(), B.data(), C.data(), m, n);
        }
        
        // Calculate and print only the average time
        double average_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / NUM_ITERATIONS;
        std::cout << average_time << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}