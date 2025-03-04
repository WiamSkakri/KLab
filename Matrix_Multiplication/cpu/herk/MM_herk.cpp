#include <mkl.h>
#include <complex>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <numeric>
using namespace std;
using namespace std::chrono;
int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " n k\n";
        return 1;
    }
    // Parse command line arguments
    MKL_INT n = atoi(argv[1]);
    MKL_INT k = atoi(argv[2]);
    if (n < 1 || k < 1) {
        cout << "Error: Matrix dimensions must be positive\n";
        return 1;
    }
    // Allocate matrices
    vector<complex<double>> A(n * k);
    vector<complex<double>> C(n * n, complex<double>(0.0, 0.0));
    // Initialize matrix A
    for (MKL_INT i = 0; i < n; i++) {
        for (MKL_INT j = 0; j < k; j++) {
            A[i + j * n] = complex<double>(
                (double)(i + 1) / (j + 1),
                (double)(i + j) / n
            );
        }
    }
    // Parameters for zherk
    const int NUM_ITERATIONS = 50;
    vector<double> execution_times;
    execution_times.reserve(NUM_ITERATIONS);
    // Warm-up run
    cblas_zherk(
        CblasColMajor, CblasUpper, CblasNoTrans,
        n, k, 1.0, A.data(), n,
        0.0, C.data(), n
    );
    // Perform 50 iterations
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        fill(C.begin(), C.end(), complex<double>(0.0, 0.0));
        auto start = high_resolution_clock::now();
        cblas_zherk(
            CblasColMajor, CblasUpper, CblasNoTrans,
            n, k, 1.0, A.data(), n,
            0.0, C.data(), n
        );
        auto end = high_resolution_clock::now();
        execution_times.push_back(duration_cast<microseconds>(end - start).count());
    }
    // Output only the average execution time
    cout << accumulate(execution_times.begin(), execution_times.end(), 0.0) / NUM_ITERATIONS << endl;
    return 0;
}