/////////////////////////////////////////////////////////////////////////////
/////   nvcc -O2 -arch=sm_75 -o fft_cuda_shared.out fft_cuda_shared.cu  /////
/////////////////////////////////////////////////////////////////////////////

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
static void fft_rows_shared(cuDoubleComplex *d_mat, unsigned int n, unsigned int n_bits)
{
    // unsigned int (n / 2) = n / 2;

    // Turing hard limit: 1024 threads/block
    unsigned int rows_per_block = 1024 / (n / 2);
    if (rows_per_block < 1){
        rows_per_block = 1;
    }

    // T4 supports up to 64KB of dynamic shared memory per block
    const size_t max_shared = 64 * 1024;
    while (rows_per_block > 1 && rows_per_block * n * sizeof(cuDoubleComplex) > max_shared){
        rows_per_block /= 2;
    }

    size_t shmem_bytes = (size_t)rows_per_block * n * sizeof(cuDoubleComplex);

    dim3 block((n / 2), rows_per_block);
    dim3 grid(n / rows_per_block);

    kernel_fft_row_shared<<<grid, block, shmem_bytes>>>(d_mat, n, n_bits, rows_per_block);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------- //
// Fallback: original global-memory implementation. Used when a           //
// row's butterfly needs more than 1024 threads                           //
// which won't fit the single-block-per-row-group scheme above            //
// ---------------------------------------------------------------------- //
static void fft_rows_global(cuDoubleComplex *d_mat, unsigned int n, unsigned int n_bits){
    int THREADS_PER_BLOCK = 256;
    dim3 n_blocks_br(n / THREADS_PER_BLOCK, n);
    kernel_bit_reverse<<<n_blocks_br, THREADS_PER_BLOCK>>>(d_mat, n, n_bits);
    cudaDeviceSynchronize();

    dim3 n_blocks_bf((n / 2) / THREADS_PER_BLOCK, n);
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m = 1u << s;
        unsigned int step = n / m;
        kernel_butterfly<<<n_blocks_bf, THREADS_PER_BLOCK>>>(d_mat, n, m, step);
        cudaDeviceSynchronize();
    }
}

// Picks the fused shared-memory path whenever it fits in one block,
// otherwise falls back to the global-memory multi-launch version.
static void fft_rows(cuDoubleComplex *d_mat, unsigned int n, unsigned int n_bits){
    if ((n / 2) <= 1024) {
        fft_rows_shared(d_mat, n, n_bits);
    } else {
        fft_rows_global(d_mat, n, n_bits);
    }
}

// ---------------------------------------------------------------------- //
// Main                                                                   //
// ---------------------------------------------------------------------- //
int main(int argc, char *argv[]){
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N> [threads_per_block_for_tuning]\n", argv[0]);
        return 1;
    }

    int N  = atoi(argv[1]);

    const unsigned int n = N;
    const unsigned int n_bits = (unsigned int)log2((double)n);
    const size_t size = n * n * sizeof(cuDoubleComplex);
    const size_t fx = 2, fy = 3;

    const int TILE = 32; 

    // ---------------------------------------------------------------------- //
    // Timing                                                                 //
    // ---------------------------------------------------------------------- //
    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t HtD_start, HtD_stop;
    cudaEvent_t DtH_start, DtH_stop;

    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventCreate(&HtD_start);
    cudaEventCreate(&HtD_stop);
    cudaEventCreate(&DtH_start);
    cudaEventCreate(&DtH_stop);

    // ---------------------------------------------------------------------- //
    // Host allocation and dataset creation                                   //
    // ---------------------------------------------------------------------- //
    cuDoubleComplex *h_result;
    cudaMallocHost((void**)&h_result, size);

    struct dataset   data     = create_dataset(n, fx, fy);
    cuDoubleComplex *h_Gxy;
    cudaMallocHost((void**)&h_Gxy, size);
    memcpy(h_Gxy, data.Gxy, size);

    
    // ---------------------------------------------------------------------- //
    // Device allocation                                                      //
    // ---------------------------------------------------------------------- //
    cuDoubleComplex *d_Gxy, *d_Gxy_T;
    cudaMalloc((void**)&d_Gxy,   size);
    cudaMalloc((void**)&d_Gxy_T, size);

    cudaEventRecord(HtD_start);
    cudaMemcpy(d_Gxy, h_Gxy, size, cudaMemcpyHostToDevice);
    cudaEventRecord(HtD_stop);
    cudaEventSynchronize(HtD_stop);

    float time_HtD;
    cudaEventElapsedTime(&time_HtD, HtD_start, HtD_stop);

    // ---------------------------------------------------------------------- //
    // Grid / block configs for the transpose                                 //
    // ---------------------------------------------------------------------- //
    dim3 blk_tr(TILE, 8);
    dim3 grd_tr((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    cudaEventRecord(ev_start);

    fft_rows(d_Gxy, n, n_bits);

    //   Transpose, so the same row-parallel, coalesced-access kernel can be reused
    kernel_transpose_shared<<<grd_tr, blk_tr>>>(d_Gxy, d_Gxy_T, n);
    cudaDeviceSynchronize();

    fft_rows(d_Gxy_T, n, n_bits);

    kernel_transpose_shared<<<grd_tr, blk_tr>>>(d_Gxy_T, d_Gxy, n);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, ev_start, ev_stop);

    // ---------------------------------------------------------------------- //
    // Copy result back and validate                                          //
    // ---------------------------------------------------------------------- //
    cudaEventRecord(DtH_start);
    cudaMemcpy(h_result, d_Gxy, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(DtH_stop);

    cudaEventSynchronize(DtH_stop);

    float time_DtH;
    cudaEventElapsedTime(&time_DtH, DtH_start, DtH_stop);

    float total_time;
    cudaEventElapsedTime(&total_time, HtD_start, DtH_stop);

    validate_fft(h_result, n, fx, fy);

    printf("%.9f\t%.9f\t%.9f\t%.9f\n", total_time * 1e-3, elapsed * 1e-3, time_HtD * 1e-3, time_DtH * 1e-3);

    cudaFree(d_Gxy);
    cudaFree(d_Gxy_T);
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