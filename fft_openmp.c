///////////////////////////////////////////////////////////////////
/////	gcc fft_openmp.c -o fft_openmp.out -lm -fopenmp		  /////
///////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <omp.h>

#include "utils_openmp.c"

int main(int argc, char *argv[]) {

    size_t N  = atoi(argv[1]);
    size_t nth = atoi(argv[2]);
    size_t fx = 2;
    size_t fy = 3;

    // set number of threads from command line
    omp_set_num_threads(nth);

    double complex *X_kl   = (double complex *)malloc(N * N * sizeof(double complex));
    double *abs_v  = (double *)malloc(N * N * sizeof(double));
    double *grid_f = (double *)malloc(N * sizeof(double));
    double *grid_f_shift = (double *)malloc(N * sizeof(double));

    struct timespec start, t1;
    double elapsed = 0.0;

    struct dataset data = create_dataset(N, fx, fy);

    clock_gettime(CLOCK_MONOTONIC, &start);

    // Malloc inside the parallel block so every
    // thread gets a private copy, then free it before the block closes.
    #pragma omp parallel
    {
        double complex *X_k = (double complex *)malloc(N * sizeof(double complex));

        // schedule(type): Specifies how iterations of the for loop are divided among available threads
        // type = static: Iterations of a loop are divided into chunks of size ceiling(number_of_iterations/number_of_threads).
        // Each thread is assigned a separate chunk.
        // dynamic Costs a small synchronisation overhead per chunk
        // In these loops, every iteration does exactly the same amount of work
        #pragma omp for schedule(static)    
        for (int i = 0; i < (int)N; i++) {
            iterative_fft(data.Gxy + i * N, X_k, N);

            // Write back in-place: row i only, no overlap between threads
            for (int j = 0; j < (int)N; j++) {
                data.Gxy[i * N + j] = X_k[j];
            }
        }

        free(X_k);
    }

    // Implicit barrier at end of parallel block:
    // all rows are fully written before column pass starts.

    // Columns are not contiguous in memory (stride = N), so copy
    // each column into a temporary buffer before calling iterative_fft.
    // Each thread handles different columns
    #pragma omp parallel
    {
        double complex *col = (double complex *)malloc(N * sizeof(double complex));
        double complex *X_l = (double complex *)malloc(N * sizeof(double complex));

        #pragma omp for schedule(static)
        for (int i = 0; i < (int)N; i++) {
            for (int j = 0; j < (int)N; j++) {
                col[j] = data.Gxy[i + j * N];
            }

            iterative_fft(col, X_l, N);

            for (int j = 0; j < (int)N; j++) {
                X_kl[i * N + j] = X_l[j];
            }
        }

        free(col);
        free(X_l);
    }

    // The above step stores the transpose results, so transpose back
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)N; i++) {
        for (int j = i + 1; j < (int)N; j++) {
            double complex tmp   = X_kl[i * N + j];
            X_kl[i * N + j]      = X_kl[j * N + i];
            X_kl[j * N + i]      = tmp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    elapsed = get_elapsed_time(start, t1);
    printf("%.6f\n", elapsed);

    for (int i = 0; i < (int)N; i++) {
        grid_f[i] = i;
    }

    for (int i = 0; i < (int)N; i++) {
        grid_f_shift[i] = (double)(i - (int)N/2);
    }

    if (N == 8) {
        create_json("fft_res_omp.json", grid_f, grid_f, X_kl, N);
    }

    validate_fft(X_kl, N, fx, fy);

    free(data.grid_x);
    free(data.grid_y);
    free(data.Gxy);
    free(grid_f);
    free(X_kl);
    free(abs_v);
    return 0;
}