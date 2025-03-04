
#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
// Function to get current time in microseconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}
// Function to print usage information
void print_usage(const char* program_name) {
    printf("Usage: %s M N K\n", program_name);
    printf("  M: Number of rows in matrix A and C\n");
    printf("  N: Number of columns in matrix B and C\n");
    printf("  K: Number of columns in A and rows in B\n");
}
// Function to initialize matrices with random values
void initialize_matrix(MKL_F16* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (MKL_F16)((double)rand() / RAND_MAX);
    }
}
int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }
    // Parse dimensions
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);
    const int iterations = 50;  // Hardcoded number of iterations
    // Validate input
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "Error: Dimensions must be positive integers\n");
          return 1;
    }
    // Allocate memory with error checking
    MKL_F16 *A = (MKL_F16*)malloc(M * K * sizeof(MKL_F16));
    MKL_F16 *B = (MKL_F16*)malloc(K * N * sizeof(MKL_F16));
    MKL_F16 *C = (MKL_F16*)malloc(M * N * sizeof(MKL_F16));
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }
    // Initialize matrices with random values
    initialize_matrix(A, M, K);
    initialize_matrix(B, K, N);
    memset(C, 0, M * N * sizeof(MKL_F16));
    // Parameters for matrix multiplication
    double total_time = 0.0;
    MKL_F16 alpha = 1.0;
    MKL_F16 beta = 0.0;
    // Perform warm-up iteration
    cblas_hgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                alpha,
                A, M,
                B, K,
                beta,
                C, M);
    // Main computation loop
    for (int i = 0; i < iterations; i++) {
        double start = get_time();
        cblas_hgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    alpha,
                    A, M,
                    B, K,
                    beta,
                    C, M);
        double end = get_time();
        total_time += (end - start);
    }
    // Calculate and print only the average time in milliseconds
    double avg_time_ms = (total_time / iterations) / 1000.0;
    printf("Average time: %.2f ms\n", avg_time_ms);
    // Cleanup
    free(A);
    free(B);
    free(C);
    return 0;
}