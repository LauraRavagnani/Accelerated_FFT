#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>

#include "utils_prova.cu"

#define N 512

int main()
{
    const unsigned int n = N;

    const unsigned int n_bits = (unsigned int)log2((double)n);

    const size_t size = n*n*sizeof(cuDoubleComplex);

    const size_t fx = 2;
    const size_t fy = 2;

    const int TILE = 32;

    cuDoubleComplex *h_result = (cuDoubleComplex*)malloc(size);

    struct dataset data = create_dataset(n, fx, fy);

    cuDoubleComplex *d_A;
    cuDoubleComplex *d_AT;

    cudaMalloc(&d_A,size);
    cudaMalloc(&d_AT,size);

    cudaMemcpy(d_A, data.Gxy, size, cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // ==========================================================
    // Row FFT
    // ==========================================================

    fft_rows(d_A, n, n_bits);

    // ==========================================================
    // Transpose
    // ==========================================================

    dim3 block2d(TILE, TILE);

    dim3 grid2d(n/TILE, n/TILE);

    kernel_transpose<<<grid2d, block2d>>>(d_A, d_AT, n);

    cudaDeviceSynchronize();

    // ==========================================================
    // Column FFT
    // ==========================================================

    fft_rows(d_AT, n, n_bits);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;

    cudaEventElapsedTime(&elapsed, start, stop);

    cudaMemcpy(
        h_result,
        d_AT,
        size,
        cudaMemcpyDeviceToHost
    );

    validate_fft(
        h_result,
        n,
        fx,
        fy
    );

    printf("%.9f\n",
           elapsed*1e-3);

    cudaFree(d_A);
    cudaFree(d_AT);

    free(h_result);
    free(data.grid_x);
    free(data.grid_y);
    free(data.Gxy);

    return 0;
}