// =============================================================================
// main_smem.cu
// 2D FFT — shared memory version, exactly 2 kernels
//
// Algorithm (row-column decomposition):
//   1. kernel_fft_row_smem  on every row of G       (1 launch)
//   2. kernel_transpose     G → G_T                 (1 launch)
//   3. kernel_fft_row_smem  on every row of G_T     (1 launch)
//
// Total: 3 kernel launches (vs 21 in the global memory version)
//
// Shared memory per block for kernel_fft_row_smem:
//   N * sizeof(cuDoubleComplex) = 512 * 16 = 8192 bytes  (well within 48 KB)
//
// blockDim for kernel_fft_row_smem is fixed at N/2 = 256 (not a CLI argument)
// because the kernel requires exactly one thread per butterfly pair.
//
// Compile:
//   nvcc -O2 -arch=sm_53 -o fft2d_smem main_smem.cu
// Run:
//   ./fft2d_smem
// =============================================================================

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils_cuda_shared.cu"

#define N 512   // must be a power of two
                // constraint: N * 16 bytes must fit in 48 KB smem → N ≤ 3072
#define TILE 32

int main(void)
{
    const unsigned int n      = N;
    const unsigned int n_bits = (unsigned int)log2((double)n);
    const size_t       size   = n * n * sizeof(cuDoubleComplex);
    const size_t       fx = 2, fy = 2;

    // -------------------------------------------------------------------------
    // Verify shared memory requirement at runtime
    // -------------------------------------------------------------------------
    size_t smem_bytes = n * sizeof(cuDoubleComplex);   // bytes per block
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (smem_bytes > prop.sharedMemPerBlock) {
            fprintf(stderr,
                "N=%u requires %zu bytes smem per block but device has %zu.\n"
                "Reduce N or switch to the global memory version.\n",
                n, smem_bytes, prop.sharedMemPerBlock);
            return 1;
        }
        printf("Device : %s\n", prop.name);
        printf("smem/block used : %zu / %zu bytes\n",
               smem_bytes, prop.sharedMemPerBlock);
    }

    // -------------------------------------------------------------------------
    // Launch configuration
    //
    // kernel_fft_row_smem:
    //   blockDim.x = n/2   (one thread per butterfly pair — fixed by algorithm)
    //   gridDim.x  = n     (one block per row)
    //
    // kernel_transpose:
    //   blockDim = (TILE, TILE) = (32, 32)
    //   gridDim  = (n/TILE, n/TILE)
    // -------------------------------------------------------------------------
    const unsigned int TPB_FFT = n / 2;       // = 256 for N=512
    //const int          TILE    = 32;

    dim3 blk_fft(TPB_FFT);
    dim3 grd_fft(n);                          // one block per row

    dim3 blk_tr(TILE, TILE);
    dim3 grd_tr(n / TILE, n / TILE);

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

    cudaMemcpy(d_A, data.Gxy, size, cudaMemcpyHostToDevice);

    // -------------------------------------------------------------------------
    // Timing
    // -------------------------------------------------------------------------
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    // =========================================================================
    // STEP 1 — FFT every row (bit-reverse + all stages inside smem, 1 launch)
    // =========================================================================
    kernel_fft_row_smem<<<grd_fft, blk_fft, smem_bytes>>>(d_A, n, n_bits);
    cudaDeviceSynchronize();

    // =========================================================================
    // STEP 2 — Tiled transpose (coalesced read + coalesced write via smem tile)
    // =========================================================================
    kernel_transpose<<<grd_tr, blk_tr>>>(d_A, d_A_T, n);
    cudaDeviceSynchronize();

    // =========================================================================
    // STEP 3 — FFT every row of transposed matrix (= column FFTs of original)
    // =========================================================================
    kernel_fft_row_smem<<<grd_fft, blk_fft, smem_bytes>>>(d_A_T, n, n_bits);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    printf("Elapsed : %.9f s\n", (double)ms * 1e-3);

    // -------------------------------------------------------------------------
    // Copy result and validate
    // -------------------------------------------------------------------------
    cudaMemcpy(h_result, d_A_T, size, cudaMemcpyDeviceToHost);
    validate_fft(h_result, n, fx, fy);

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
