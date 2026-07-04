// =============================================================================
// Algorithm (row-column decomposition), shared-memory version:
//   1. kernel_fft_row_shared  : bit-reverse + all butterfly stages for every
//                               row of G, fused into ONE kernel, row kept
//                               resident in shared memory start to finish.
//   2. kernel_transpose_shared: tiled shared-memory transpose  G  ->  G_T
//   3. kernel_fft_row_shared  : same fused FFT on every row of G_T
//                               (= every column of G)
//
// Why this is faster on a T4 (Turing, sm_75, 40 SMs, 64KB shared mem/SM):
//   The original code did 1 bit-reverse launch + log2(n) butterfly launches
//   per FFT pass, each one reading the ENTIRE row from global memory and
//   writing it back. For n=512 that's 10 extra global-memory round trips
//   per row, all serialized by cudaDeviceSynchronize() between them, plus
//   kernel-launch overhead ~10 times over. Since each row's FFT only
//   depends on data within that row, the whole thing fits in shared memory
//   (n * 16 bytes; e.g. 8KB for n=512, 32KB for n=2048), so it can be done
//   with exactly one global read and one global write per row, with all
//   intermediate stages just doing __syncthreads() on fast on-chip memory.
//   The transpose gets the same treatment: a padded tile (33 columns
//   instead of 32) removes shared-memory bank conflicts, and strided
//   BLOCK_ROWS loads keep every global access coalesced.
//
// Compile:
//   nvcc -O2 -arch=sm_75 -o fft_cuda.out fft_cuda.cu
// =============================================================================

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils_cuda_shared.cu"

// -----------------------------------------------------------------------------
// Fused shared-memory row-FFT pass. One kernel launch does the bit-reversal
// and every Cooley-Tukey stage for all n rows of d_mat.
// -----------------------------------------------------------------------------
static void fft_rows_shared(cuDoubleComplex *d_mat,
                            unsigned int     n,
                            unsigned int     n_bits)
{
    unsigned int half_n = n / 2;

    // Turing (sm_75) hard limit: 1024 threads/block.
    unsigned int rows_per_block = 1024 / half_n;
    if (rows_per_block < 1) rows_per_block = 1;

    // T4 supports up to 64KB of dynamic shared memory per block (opt-in).
    const size_t max_shared = 64 * 1024;
    while (rows_per_block > 1 &&
           (size_t)rows_per_block * n * sizeof(cuDoubleComplex) > max_shared)
        rows_per_block /= 2;

    // keep the grid clean: rows_per_block must divide n
    while (rows_per_block > 1 && (n % rows_per_block) != 0)
        rows_per_block /= 2;

    size_t shmem_bytes = (size_t)rows_per_block * n * sizeof(cuDoubleComplex);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(kernel_fft_row_shared,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              (int)max_shared);
        attr_set = true;
    }

    dim3 block(half_n, rows_per_block);
    dim3 grid(n / rows_per_block);

    kernel_fft_row_shared<<<grid, block, shmem_bytes>>>(d_mat, n, n_bits, rows_per_block);
    cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------
// Fallback: original global-memory multi-launch implementation. Used when a
// row's butterfly needs more than 1024 threads (n/2 > 1024, i.e. n > 2048),
// which won't fit the single-block-per-row-group scheme above, and used by
// the tuning path (-tune) to sweep THREADS_PER_BLOCK as before.
// -----------------------------------------------------------------------------
static void fft_rows_global(cuDoubleComplex *d_mat,
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
        kernel_butterfly<<<n_blocks_bf, THREADS_PER_BLOCK>>>(d_mat, n, m, step);
        cudaDeviceSynchronize();
    }
}

static void fft_rows_tune(cuDoubleComplex *d_mat,
                     unsigned int     n,
                     unsigned int     n_bits,
		            int THREADS_PER_BLOCK)
{
    dim3 n_blocks_br(n / THREADS_PER_BLOCK, n);
    kernel_bit_reverse<<<n_blocks_br, THREADS_PER_BLOCK>>>(d_mat, n, n_bits);
    cudaDeviceSynchronize();

    dim3 n_blocks_bf((n / 2) / THREADS_PER_BLOCK, n);
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m    = 1u << s;
        unsigned int step = n / m;
        kernel_butterfly<<<n_blocks_bf, THREADS_PER_BLOCK>>>(d_mat, n, m, step);
        cudaDeviceSynchronize();
    }
}

// Picks the fused shared-memory path whenever it fits in one block,
// otherwise falls back to the global-memory multi-launch version.
static void fft_rows(cuDoubleComplex *d_mat,
                     unsigned int     n,
                     unsigned int     n_bits)
{
    unsigned int half_n = n / 2;
    if (half_n <= 1024) {
        fft_rows_shared(d_mat, n, n_bits);
    } else {
        fft_rows_global(d_mat, n, n_bits);
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    bool tune = false;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N> [threads_per_block_for_tuning]\n", argv[0]);
        return 1;
    }

    int N  = atoi(argv[1]);
    if (argc >= 3) tune = true;

    const unsigned int n      = N;
    const unsigned int n_bits = (unsigned int)log2((double)n);
    const size_t       size   = n * n * sizeof(cuDoubleComplex);
    const size_t       fx = 2, fy = 2;

    const int TILE = 32; // matches TR_TILE_DIM in utils_cuda.cu

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
    cuDoubleComplex *h_result;
    cudaMallocHost((void**)&h_result, size);

    struct dataset   data     = create_dataset(n, fx, fy);
    cuDoubleComplex *h_Gxy;
    cudaMallocHost((void**)&h_Gxy, size);
    memcpy(h_Gxy, data.Gxy, size);

    // -------------------------------------------------------------------------
    // Device allocation
    // -------------------------------------------------------------------------
    cuDoubleComplex *d_A, *d_A_T;
    cudaMalloc((void**)&d_A,   size);
    cudaMalloc((void**)&d_A_T, size);

    cudaEventRecord(HtD_start);
    cudaMemcpy(d_A, h_Gxy, size, cudaMemcpyHostToDevice);
    cudaEventRecord(HtD_stop);
    cudaEventSynchronize(HtD_stop);

    float time_HtD;
    cudaEventElapsedTime(&time_HtD, HtD_start, HtD_stop);

    // -------------------------------------------------------------------------
    // Grid / block configs for the transpose (block matches TR_BLOCK_ROWS)
    // -------------------------------------------------------------------------
    dim3 blk_tr(TILE, 8);
    dim3 grd_tr((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    cudaEventRecord(ev_start);

    // =========================================================================
    // STEP 1 — fused FFT every row  (kernel_fft_row_shared, shared memory)
    // =========================================================================
    if (tune) {
        int THREADS_PER_BLOCK = atoi(argv[2]);
        fft_rows_tune(d_A, n, n_bits, THREADS_PER_BLOCK);
    } else {
        fft_rows(d_A, n, n_bits);
    }

    // =========================================================================
    // STEP 2 — Transpose  (kernel_transpose_shared, tiled + padded)
    // =========================================================================
    kernel_transpose_shared<<<grd_tr, blk_tr>>>(d_A, d_A_T, n);
    cudaDeviceSynchronize();

    // =========================================================================
    // STEP 3 — fused FFT every row of transposed matrix
    // =========================================================================
    if (tune) {
        int THREADS_PER_BLOCK = atoi(argv[2]);
        fft_rows_tune(d_A_T, n, n_bits, THREADS_PER_BLOCK);
    } else {
        fft_rows(d_A_T, n, n_bits);
    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, ev_start, ev_stop);

    // -------------------------------------------------------------------------
    // Copy result back and validate
    // -------------------------------------------------------------------------
    cudaEventRecord(DtH_start);
    cudaMemcpy(h_result, d_A_T, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(DtH_stop);

    cudaEventSynchronize(DtH_stop);

    float time_DtH;
    cudaEventElapsedTime(&time_DtH, DtH_start, DtH_stop);

    float total_time;
    cudaEventElapsedTime(&total_time, HtD_start, DtH_stop);

    validate_fft(h_result, n, fx, fy);

    printf("%.9f\t%.9f\t%.9f\t%.9f\n", total_time * 1e-3, elapsed * 1e-3, time_HtD * 1e-3, time_DtH * 1e-3);

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
    cudaFreeHost(h_result);
    free(data.grid_x);
    free(data.grid_y);
    free(data.Gxy);
    cudaFreeHost(h_Gxy);

    return 0;
}