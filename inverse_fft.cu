// =============================================================================
// Algorithm (row-column decomposition, inverse transform):
//   1. kernel_bit_reverse       on every row of X   (frequency-domain input)
//   2. kernel_butterfly_inverse on every row of X    (log2 N stages)
//   3. kernel_transpose         X  →  X_T
//   4. kernel_bit_reverse       on every row of X_T   (= every column of X)
//   5. kernel_butterfly_inverse on every row of X_T   (log2 N stages)
//   6. kernel_scale             X_T /= (N*N)          (2-D IDFT normalization)
//
// Compile:
//   nvcc -O2 -arch=sm_75 -o inverse_fft.out inverse_fft.cu
// =============================================================================

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils_inverse_fft.cu"

// #define N 512   // must be a power of two
// #define THREADS_PER_BLOCK 128    // COMMENT WHEN TUNING


// -----------------------------------------------------------------------------
// Run one full 1-D IFFT pass over every row of d_mat
//   - one kernel_bit_reverse launch
//   - log2(n) kernel_butterfly_inverse launches
//   - one cudaDeviceSynchronize() at the end of the butterfly loop
// -----------------------------------------------------------------------------
static void ifft_rows(cuDoubleComplex *d_mat,
                      unsigned int     n,
                      unsigned int     n_bits)
{
    int THREADS_PER_BLOCK = 256;
    dim3 n_blocks_br(n / THREADS_PER_BLOCK, n);
    kernel_bit_reverse<<<n_blocks_br, THREADS_PER_BLOCK>>>(d_mat, n, n_bits);
    cudaDeviceSynchronize();

    dim3 n_blocks_bf((n / 2) / THREADS_PER_BLOCK, n);
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m    = 1u << s;
        unsigned int step = n / m;
        kernel_butterfly_inverse<<<n_blocks_bf, THREADS_PER_BLOCK>>>(d_mat, n, m, step);
        cudaDeviceSynchronize();
    }
}


// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char *argv[])
{

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <threads_per_block>\n", argv[0]);
        return 1;
    }

    int N  = atoi(argv[1]);

    const unsigned int n      = N;
    const unsigned int n_bits = (unsigned int)log2((double)n);
    const size_t       size   = n * n * sizeof(cuDoubleComplex);
    const size_t       fx = 2, fy = 2;

    // tile size for the transpose kernel (keep ≤ 32 so block ≤ 1024 threads)
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
    // Host allocation and dataset creation (frequency-domain input)
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
    // Grid / block configs for the transpose and normalization
    // -------------------------------------------------------------------------
    dim3 blk_tr(TILE, TILE);
    dim3 grd_tr(n / TILE, n / TILE);

    int THREADS_PER_BLOCK_SCALE = 256;
    dim3 n_blocks_scale(n / THREADS_PER_BLOCK_SCALE, n);

    cudaEventRecord(ev_start);

    // =========================================================================
    // STEP 1 — IFFT every row  (kernel_bit_reverse + kernel_butterfly_inverse)
    // =========================================================================
    
    ifft_rows(d_A, n, n_bits);


    // =========================================================================
    // STEP 2 — Transpose  (kernel_transpose)
    // =========================================================================
    kernel_transpose<<<grd_tr, blk_tr>>>(d_A, d_A_T, n);
    cudaDeviceSynchronize();

    // =========================================================================
    // STEP 3 — IFFT every row of transposed matrix
    // =========================================================================

    ifft_rows(d_A_T, n, n_bits);
    
    // =========================================================================
    // STEP 4 — Normalize  (kernel_scale: divide every element by n*n)
    // =========================================================================
    kernel_scale<<<n_blocks_scale, THREADS_PER_BLOCK_SCALE>>>(d_A_T, n);
    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------
    // Copy result back and validate
    // -------------------------------------------------------------------------
    cudaMemcpy(h_result, d_A_T, size, cudaMemcpyDeviceToHost);

    validate_ifft(h_result, n, fx, fy);

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    cudaFree(d_A);
    cudaFree(d_A_T);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaEventDestroy(DtH_start);
    cudaEventDestroy(DtH_stop);
    cudaEventDestroy(HtD_start);
    cudaEventDestroy(HtD_stop);
    free(h_result);
    free(data.grid_x);
    free(data.grid_y);
    free(data.Gxy);

    return 0;
}
