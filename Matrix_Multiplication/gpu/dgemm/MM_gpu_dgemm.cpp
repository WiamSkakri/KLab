#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>

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

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    // Parse dimensions from command line
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    // Allocate host memory
    double *h_A = (double *)malloc(m * k * sizeof(double));
    double *h_B = (double *)malloc(k * n * sizeof(double));
    double *h_C = (double *)malloc(m * n * sizeof(double));

    // Initialize matrices with random values
    for (int i = 0; i < m * k; i++) {
        h_A[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        h_B[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < m * n; i++) {
        h_C[i] = 0.0;
    }

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(double)));

    // Copy input matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, m * n * sizeof(double)));

    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Parameters for GEMM
    const double alpha = 1.0;
    const double beta = 0.0;

    // Perform 50 iterations and measure time
    double total_time = 0.0;
    for (int iter = 0; iter < 50; iter++) {
        // Reset device matrix C to zero
        CUDA_CHECK(cudaMemset(d_C, 0, m * n * sizeof(double)));

        double start_time = get_time();

        // Perform matrix multiplication
        CUBLAS_CHECK(cublasDgemm(handle, 
                                  CUBLAS_OP_N, CUBLAS_OP_N, 
                                  n, m, k, 
                                  &alpha, 
                                  d_B, n, 
                                  d_A, k, 
                                  &beta, 
                                  d_C, n));

        // Synchronize to ensure completion
        CUDA_CHECK(cudaDeviceSynchronize());

        double end_time = get_time();
        total_time += (end_time - start_time);
    }

    // Optional: Copy result back to host (if needed)
    // CUDA_CHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    printf("%d,%d,%d,%.6f\n", m, n, k, total_time / 50.0);

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}