#include <mkl.h>
#include <complex>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <cstdlib>
#include <ctime>

void initialize_hermitian_matrix(std::complex<double>* A, MKL_INT m) {
    srand(time(NULL));
    for (MKL_INT i = 0; i < m; i++) {
        for (MKL_INT j = i; j < m; j++) {
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

void initialize_matrix(std::complex<double>* matrix, MKL_INT rows, MKL_INT cols) {
    for (MKL_INT i = 0; i < rows * cols; i++) {
        double real = (double)rand()/RAND_MAX;
        double imag = (double)rand()/RAND_MAX;
        matrix[i] = std::complex<double>(real, imag);
    } 
}            

double run_single_iteration(std::complex<double>* A, std::complex<double>* B,
                          std::complex<double>* C, MKL_INT m, MKL_INT n) {
    std::complex<double> alpha(1.0, 0.0);
    std::complex<double> beta(0.0, 0.0);
    const MKL_INT lda = m;
    const MKL_INT ldb = m; 
    const MKL_INT ldc = m;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();

    cblas_zhemm(
        CblasColMajor,
        CblasLeft,
        CblasUpper,
        m,
        n,
        &alpha,
        A,
        lda,
        B,
        ldb,
        &beta,
        C,
        ldc
    );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 3) return 1;

        const MKL_INT m = std::atoi(argv[1]);
        const MKL_INT n = std::atoi(argv[2]);
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
    catch (const std::exception&) {
        return 1;
    }
}