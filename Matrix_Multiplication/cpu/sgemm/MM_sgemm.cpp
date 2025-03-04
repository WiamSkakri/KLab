#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mkl.h>
#include <iomanip>
// Enum to specify whether to transpose a matrix (if needed in future extensions)
enum class Transpose {
    No,
    Yes
};
/**
 * @brief Performs matrix multiplication using MKL's cblas_sgemm and measures execution time.
 * 
 * @param m Number of rows in Matrix A and Matrix C.
 * @param n Number of columns in Matrix B and Matrix C.
 * @param k Number of columns in Matrix A and rows in Matrix B.
 * @param iterations Number of multiplication iterations to perform.
 * @return double Average execution time in seconds over the specified iterations.
 */
double perform_gemm(int m, int n, int k, int iterations = 50) {
    // Leading dimensions for row-major layout
    MKL_INT lda = k;
    MKL_INT ldb = n;
    MKL_INT ldc = n;
    // Scalars for GEMM
    float alpha = 1.0f;
    float beta = 0.0f;
    // Allocate matrices using std::vector for automatic memory management
    std::vector<float> A(m * k, 0.0f);
    std::vector<float> B(k * n, 0.0f);
    std::vector<float> C(m * n, 0.0f);
    // Random number generation setup
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    // Total duration accumulator
    std::chrono::duration<double> total_duration = std::chrono::duration<double>::zero();
 // Total duration accumulator
    std::chrono::duration<double> total_duration = std::chrono::duration<double>::zero();
    // Warm-up iteration (optional)
    for (int warm = 0; warm < 2; ++warm) {
        // Initialize matrices A and B with random numbers
        for (auto &val : A) val = dist(rng);
        for (auto &val : B) val = dist(rng);
        // Perform GEMM
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    }
    // Main timing loop
    for (int iteration = 0; iteration < iterations; ++iteration) {
        // Initialize matrices A and B with random numbers
        for (auto &val : A) val = dist(rng);
        for (auto &val : B) val = dist(rng);
        // Optionally, reset matrix C to zero if beta is not zero
        // Not necessary here since beta = 0.0f
        // Start timer
        auto start = std::chrono::high_resolution_clock::now();
        // Perform matrix multiplication using cblas_sgemm
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
        // End timer
        auto end = std::chrono::high_resolution_clock::now();
        // Accumulate duration
        total_duration += (end - start);
    }
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
    // Optionally, set the number of MKL threads (uncomment to set)
    // mkl_set_num_threads(4); // Example: set to 4 threads
    // Perform GEMM and measure average execution time
    double avg_time = perform_gemm(m, n, k);
    // Display the results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Average execution time over 50 iterations: " << avg_time << " seconds\n";
    return EXIT_SUCCESS;
}