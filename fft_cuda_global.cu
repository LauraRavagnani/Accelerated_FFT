// =============================================================================
// main.cu
// 2D FFT — global memory only, exactly 3 kernels
//
// Algorithm (row-column decomposition):
//   1. kernel_bit_reverse  on every row of G
//   2. kernel_butterfly    on every row of G  (log2 N stages)
//   3. kernel_transpose    G  →  G_T
//   4. kernel_bit_reverse  on every row of G_T   (= every column of G)
//   5. kernel_butterfly    on every row of G_T   (log2 N stages)
//   (optional 6. kernel_transpose G_T → G  if you want row-major output)
//
// Compile:
//   nvcc -O2 -arch=sm_53 -o fft2d main.cu
// Run:
//   ./fft2d <threads_per_block>        e.g.   ./fft2d 128
// =============================================================================

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils_cuda_global.cu"

#define N 512   // must be a power of two
#define THREADS_PER_BLOCK 128

// -----------------------------------------------------------------------------
// Helper: run one full 1-D FFT pass over every row of d_mat
//   - one kernel_bit_reverse launch
//   - log2(n) kernel_butterfly launches
//   - one cudaDeviceSynchronize() at the end of the butterfly loop
// -----------------------------------------------------------------------------
static void fft_rows(cuDoubleComplex *d_mat,
                     unsigned int     n,
                     unsigned int     n_bits,
                     int             THREADS_PER_BLOCK)
{
    // bit-reverse: each thread handles one element, 2D grid over all rows
    //dim3 blk_br(THREADS_PER_BLOCK);
    dim3 n_blocks_br(n / THREADS_PER_BLOCK, n);
    kernel_bit_reverse<<<n_blocks_br, THREADS_PER_BLOCK>>>(d_mat, n, n_bits);
    cudaDeviceSynchronize();

    // butterfly stages: each thread handles one pair, 2D grid over all rows
    //dim3 blk_bf(THREADS_PER_BLOCK);
    dim3 n_blocks_bf((n / 2) / THREADS_PER_BLOCK, n);
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m    = 1u << s;
        unsigned int step = n / m;
        kernel_butterfly<<<n_blocks_bf, THREADS_PER_BLOCK>>>(d_mat, n, m, step);
        cudaDeviceSynchronize();   // inter-block ordering between stages
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    // if (argc < 2) {
    //     fprintf(stderr, "Usage: %s <threads_per_block>\n", argv[0]);
    //     return 1;
    // }
    // //const int THREADS_PER_BLOCK = 512;
    // int THREADS_PER_BLOCK  = atoi(argv[1]);

    const unsigned int n      = N;
    const unsigned int n_bits = (unsigned int)log2((double)n);
    const size_t       size   = n * n * sizeof(cuDoubleComplex);
    const size_t       fx = 2, fy = 2;

    // tile size for the transpose kernel (keep ≤ 32 so block ≤ 1024 threads)
    //const int TILE = (THREADS_PER_BLOCK >= 32) ? 32 : (int)sqrt((double)THREADS_PER_BLOCK);
    const int TILE = 32;

    // -------------------------------------------------------------------------
    // Timing
    // -------------------------------------------------------------------------
    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t HtD_start, HtD_stop;
    cudaEvent_t DtH_start, DtH_stop;
    
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventCreate(&HtD_start);
    cudaEventCreate(&HtD_stop);
    cudaEventCreate(&DtH_start);
    cudaEventCreate(&DtH_stop);

    // -------------------------------------------------------------------------
    // Host allocation and dataset creation
    // -------------------------------------------------------------------------
    cuDoubleComplex *h_result = (cuDoubleComplex*)malloc(size);
    struct dataset   data     = create_dataset(n, fx, fy);

    // -------------------------------------------------------------------------
    // Device allocation
    // -------------------------------------------------------------------------
    cuDoubleComplex *d_A, *d_A_T;
    cudaMalloc((void**)&d_A,   size);
    cudaMalloc((void**)&d_A_T, size);

    cudaEventRecord(HtD_start);
    cudaMemcpy(d_A, data.Gxy, size, cudaMemcpyHostToDevice);
    cudaEventRecord(HtD_stop);
    cudaEventSynchronize(HtD_stop);

    float time_HtD;
    cudaEventElapsedTime(&time_HtD, HtD_start, HtD_stop);


    // -------------------------------------------------------------------------
    // Grid / block configs for the transpose
    // -------------------------------------------------------------------------
    dim3 blk_tr(TILE, TILE);
    dim3 grd_tr(n / TILE, n / TILE);

    cudaEventRecord(ev_start);

    // =========================================================================
    // STEP 1 — FFT every row  (kernel_bit_reverse + kernel_butterfly)
    // =========================================================================
    fft_rows(d_A, n, n_bits, THREADS_PER_BLOCK);

    // =========================================================================
    // STEP 2 — Transpose  (kernel_transpose)
    // =========================================================================
    kernel_transpose<<<grd_tr, blk_tr>>>(d_A, d_A_T, n);
    cudaDeviceSynchronize();

    // =========================================================================
    // STEP 3 — FFT every row of transposed matrix  (= column FFTs of original)
    //          (kernel_bit_reverse + kernel_butterfly)
    // =========================================================================
    fft_rows(d_A_T, n, n_bits, THREADS_PER_BLOCK);

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, ev_start, ev_stop);

    // -------------------------------------------------------------------------
    // Copy result back and validate
    // NOTE: result lives in d_A_T (transposed layout).
    //       Rows of d_A_T = columns of the 2D FFT output.
    //       Transpose back with kernel_transpose if row-major order is needed.
    // -------------------------------------------------------------------------
    cudaEventRecord(DtH_start);
    cudaMemcpy(h_result, d_A_T, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(DtH_stop);

    cudaEventSynchronize(DtH_stop);

    float time_DtH;
    cudaEventElapsedTime(&time_DtH, DtH_start, DtH_stop);
    
    validate_fft(h_result, n, fx, fy);

    printf("%.9f\t%.9f\t%.9f", elapsed * 1e-3, time_HtD * 1e-3, time_DtH * 1e-3);

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    cudaFree(d_A);
    cudaFree(d_A_T);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    free(h_result);
    free(data.grid_x);
    free(data.grid_y);
    free(data.Gxy);

    return 0;
}
