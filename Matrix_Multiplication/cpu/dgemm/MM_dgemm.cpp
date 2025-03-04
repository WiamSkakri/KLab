#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    // Parse dimensions from command line
    MKL_INT m = atoi(argv[1]);
    MKL_INT n = atoi(argv[2]);
    MKL_INT k = atoi(argv[3]);

    // Allocate matrices
    double *A = (double *)malloc(m * k * sizeof(double));
    double *B = (double *)malloc(k * n * sizeof(double));
    double *C = (double *)malloc(m * n * sizeof(double));

    // Initialize matrices with random values
    for (int i = 0; i < m * k; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < m * n; i++) {
        C[i] = 0.0;
    }

    // Parameters for GEMM
    const double alpha = 1.0;
    const double beta = 0.0;
    const MKL_INT lda = k;
    const MKL_INT ldb = n;
    const MKL_INT ldc = n;

    // Perform 50 iterations and measure time
    double total_time = 0.0;
    for (int iter = 0; iter < 50; iter++) {
        double start_time = get_time();
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m,
                    n,
                    k,
                    alpha,
                    A,
                    lda,
                    B,
                    ldb,
                    beta,
                    C,
                    ldc);
        double end_time = get_time();
        total_time += (end_time - start_time);
    }
    printf("%d,%d,%d,%.6f\n", m, n, k, total_time / 50.0);
    free(A);
    free(B);
    free(C);
    return 0;
}