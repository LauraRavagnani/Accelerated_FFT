#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils_cuda_shared.cu"

#define N 512
#define THREADS_PER_BLOCK 128

/* ------------------------------------------------------------------ */
/* Helper: FFT on every row using shared memory kernel                */
/* No tmp buffer needed — result written back in-place                */
/* ------------------------------------------------------------------ */
void cuda_fft_smem(cuDoubleComplex *d_mat,
                   cuDoubleComplex *d_tw,
                   int n, int n_bits, int TPB)
{
    size_t smem_bytes = n * sizeof(cuDoubleComplex);
    dim3 block(TPB);
    dim3 grid(n);   // one block per row

    kernel_fft_smem<<<grid, block, smem_bytes>>>(d_mat, d_tw, n, n_bits);
    cudaDeviceSynchronize();
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    int TPB = (argc > 1) ? atoi(argv[1]) : 256;

    if (TPB > N) TPB = N;

    int    n_bits = (int)log2((double)N);
    size_t size   = N * N * sizeof(cuDoubleComplex);
    size_t size_tw= (N/2) * sizeof(cuDoubleComplex);
    size_t fx = 2, fy = 2;

    dim3 block2d(TILE_DIM, TILE_DIM);
    dim3 grid2d(N / TILE_DIM, N / TILE_DIM);

    cuDoubleComplex *h_X_kl = (cuDoubleComplex*)malloc(size);
    struct dataset data = create_dataset_cuda(N, fx, fy);

    cuDoubleComplex *d_Gxy, *d_Gxy_T, *d_tw;
    cudaMalloc((void**)&d_Gxy,   size);
    cudaMalloc((void**)&d_Gxy_T, size);
    cudaMalloc((void**)&d_tw,    size_tw);

    cudaMemcpy(d_Gxy, data.Gxy, size, cudaMemcpyHostToDevice);

    int N_b_tw = (N/2 + TPB - 1) / TPB;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_precompute_twiddles<<<N_b_tw, TPB>>>(d_tw, N);
    cudaDeviceSynchronize();

    /* step 1: FFT on each row */
    cuda_fft_smem(d_Gxy, d_tw, N, n_bits, TPB);

    /* step 2: transpose */
    matrixTransposition<<<grid2d, block2d>>>(d_Gxy, d_Gxy_T, N);
    cudaDeviceSynchronize();

    /* step 3: FFT on each column (rows of transposed matrix) */
    cuda_fft_smem(d_Gxy_T, d_tw, N, n_bits, TPB);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("%.9f\n", elapsed * 1e-3);   // ms → seconds 

    cudaMemcpy(h_X_kl, d_Gxy_T, size, cudaMemcpyDeviceToHost);
    validate_fft(h_X_kl, N, fx, fy);

    cudaFree(d_Gxy);
    cudaFree(d_Gxy_T);
    cudaFree(d_tw);
    free(h_X_kl);

    return 0;
}