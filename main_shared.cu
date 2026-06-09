// =============================================================================
// main_shared.cu
// 2D FFT — shared memory, double precision, warp-divergence-free butterfly.
//
// Key design decisions vs the global-memory version:
//
//  1. SHARED MEMORY BUTTERFLY
//     Each block loads one complete row into __shared__ memory, performs all
//     log2(N) butterfly stages in-place there, then writes back once.
//     Global memory traffic: 1 read + 1 write per row (vs 2*log2(N) in the
//     global version where every stage reads and writes DRAM).
//
//  2. NO WARP DIVERGENCE IN THE BUTTERFLY
//     The classic divergence bug in FFT:
//       if (tid % m < half) { ... }   ← threads in the same warp take
//                                        different branches → divergence
//     Fix: reindex so every thread always executes the same code path.
//     We map thread index tid to butterfly pair (k, j) arithmetically:
//       j = tid % half          (position within group)
//       k = (tid / half) * m    (start of group)
//     All threads do the same operations — no branch on tid.
//     The only guard is `if (tid >= n/2) return` at the very top, which
//     only fires for out-of-range threads (padding), not mid-warp.
//     As long as TPB is a power of two and n/2 is a multiple of TPB,
//     no warp is ever split across the return boundary.
//
//  3. SHARED MEMORY TRANSPOSE (tiled)
//     The naive transpose reads columns (strided) and writes rows (coalesced).
//     The tiled version uses a TILE×(TILE+1) shared memory buffer (+1 avoids
//     bank conflicts) to convert the strided reads into coalesced ones:
//       - Each block reads a TILE×TILE tile coalesced from A into shared mem
//       - Writes a TILE×TILE tile coalesced to A_T from shared mem
//     Both directions are now coalesced → ~2× bandwidth vs naive.
//
//  4. BIT-REVERSAL WITH CONSECUTIVE ACTIVE THREADS
//     Thread tid handles element tid. The swap `if (r > tid)` means only
//     ~half the threads do work, but crucially threads 0..n/2-1 within
//     each warp are always the active ones (no interleaving of active/idle
//     threads within a warp), so no divergence penalty.
//
// Algorithm: row-column decomposition (same as global version)
//   1. bit-reverse + butterfly on every row  (in shared memory)
//   2. tiled transpose
//   3. bit-reverse + butterfly on every row of transposed matrix
//
// Compile:
//   nvcc -O2 -arch=sm_75 -o fft2d_shared main_shared.cu
// Run:
//   ./fft2d_shared <threads_per_block>    e.g.  ./fft2d_shared 512
//
// Notes:
//   - TPB must be >= N/2 so one block covers an entire row's butterfly pairs.
//     For N=512: TPB must be >= 256. Use 256 or 512.
//   - Shared memory per block = N * 16 bytes = 512*16 = 8192 bytes (well
//     within the T4's 64KB limit).
// =============================================================================

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N    512          // matrix dimension, must be power of two
#define TILE 32           // transpose tile size (32*32=1024 threads, fits T4)

// =============================================================================
// KERNEL 1 — shared memory FFT of all rows
//
// Grid  : dim3(1, N)         — one block per row
// Block : dim3(TPB)          — TPB >= N/2
//
// Each block:
//   a) loads one row from global memory into shared mem  (coalesced read)
//   b) performs in-place bit-reversal in shared mem
//   c) performs all log2(N) butterfly stages in shared mem
//   d) writes the row back to global memory               (coalesced write)
//
// Global memory traffic per row: N reads + N writes = 2*N*16 bytes
// Shared memory traffic per row: 2*log2(N)*N reads+writes (fast, on-chip)
//
// Warp divergence analysis:
//   - Loading: thread tid loads element tid → no divergence
//   - Bit-reversal: `if (r > tid)` — at most one warp straddles the boundary
//     where r==tid, and that only happens for tid=0 (r=0 always, never >0).
//     In practice zero divergence.
//   - Butterfly: no branch on tid at all. All threads compute the same
//     operations. The only guard is the top-level `if (tid >= N/2) return`
//     which fires for threads N/2..TPB-1. Since N/2=256 and TPB is 256 or
//     512, this guard fires for whole warps, not within a warp.
// =============================================================================
__global__ void kernel_fft_row_shared(cuDoubleComplex *A,
                                      unsigned int     n,
                                      unsigned int     n_bits)
{
    // Dynamic shared memory: one complex element per column
    extern __shared__ cuDoubleComplex smem[];   // size = n * sizeof(cuDoubleComplex)

    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.y;

    // ------------------------------------------------------------------
    // STEP A: load row from global memory into shared memory (coalesced)
    // Multiple loads if TPB < N (each thread loads n/TPB elements)
    // ------------------------------------------------------------------
    for (unsigned int i = tid; i < n; i += blockDim.x)
        smem[i] = A[row * n + i];
    __syncthreads();

    // ------------------------------------------------------------------
    // STEP B: in-place bit-reversal in shared memory
    // Thread tid handles element tid; swap with bit-reversed partner r.
    // Guard r > tid prevents double-swapping.
    // Consecutive threads handle consecutive elements → no divergence.
    // ------------------------------------------------------------------
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        unsigned int r = 0, tmp = i;
        for (int b = 0; b < (int)n_bits; b++) {
            r   = (r << 1) | (tmp & 1);
            tmp >>= 1;
        }
        if (r > i) {
            cuDoubleComplex t = smem[i];
            smem[i]           = smem[r];
            smem[r]           = t;
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // STEP C: Cooley-Tukey butterfly, all stages, in shared memory
    //
    // For each stage s, butterfly size m = 2^s, half = m/2.
    // Thread tid computes the pair at position (k + j, k + j + half)
    // where:
    //   j = tid % half    — position within the butterfly group
    //   k = (tid/half)*m  — start of the group
    //
    // This mapping is injective: every tid in [0, n/2) maps to a unique
    // pair, and every pair is covered exactly once. No branching on tid.
    //
    // Warp divergence: zero. All threads in a warp compute the same
    // sequence of instructions. The twiddle angle differs per thread
    // but that is data-dependent variation, not control-flow divergence.
    //
    // For TPB < n/2: each thread handles multiple pairs (stride loop).
    // For TPB >= n/2: threads tid >= n/2 are idle this step (whole warps).
    // ------------------------------------------------------------------
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m    = 1u << s;
        unsigned int half = m >> 1;
        unsigned int step = n / m;   // twiddle stride

        for (unsigned int tid2 = tid; tid2 < n / 2; tid2 += blockDim.x) {
            unsigned int j = tid2 % half;
            unsigned int k = (tid2 / half) * m;

            // Twiddle factor W_n^{j*step} = e^{-2πi * j*step / n}
            double angle = -2.0 * M_PI * (double)(j * step) / (double)n;
            double c, sv;
            sincos(angle, &sv, &c);
            cuDoubleComplex w = make_cuDoubleComplex(c, sv);

            cuDoubleComplex u = smem[k + j];
            cuDoubleComplex t = cuCmul(w, smem[k + j + half]);

            smem[k + j]        = cuCadd(u, t);
            smem[k + j + half] = cuCsub(u, t);
        }
        // Synchronize all threads in the block between stages.
        // This is correct here (unlike the global version) because all
        // threads in the block share the same smem and we need the
        // previous stage fully written before reading in the next.
        __syncthreads();
    }

    // ------------------------------------------------------------------
    // STEP D: write row back to global memory (coalesced)
    // ------------------------------------------------------------------
    for (unsigned int i = tid; i < n; i += blockDim.x)
        A[row * n + i] = smem[i];
    // No __syncthreads() needed here: each thread writes its own elements.
}

// =============================================================================
// KERNEL 2 — tiled shared memory transpose (coalesced reads AND writes)
//
// Grid  : dim3(N/TILE, N/TILE)
// Block : dim3(TILE, TILE)
//
// Naive transpose: reads column of A (strided) → writes row of A_T (coalesced)
// Tiled transpose: reads tile of A (coalesced) → shared mem → writes tile of
//                  A_T (coalesced). Both directions coalesced → ~2× bandwidth.
//
// The +1 padding on the shared memory row dimension eliminates bank conflicts:
// when threads read transposed from shared mem (column access), each thread
// hits a different bank because the stride is TILE+1, not TILE.
//
// Warp divergence: none. All threads in a warp read/write consecutive
// addresses. The boundary guard (row < n && col < n) fires for whole
// warps only when N is not a multiple of TILE (not our case at N=512).
// =============================================================================
__global__ void kernel_transpose_tiled(const cuDoubleComplex *A,
                                             cuDoubleComplex *A_T,
                                             unsigned int     n)
{
    // +1 to avoid shared memory bank conflicts on the transpose read
    __shared__ cuDoubleComplex tile[TILE][TILE + 1];

    unsigned int col = blockIdx.x * TILE + threadIdx.x;
    unsigned int row = blockIdx.y * TILE + threadIdx.y;

    // Read tile from A: threads in a warp read consecutive columns
    // → coalesced global memory read
    if (row < n && col < n)
        tile[threadIdx.y][threadIdx.x] = A[row * n + col];
    __syncthreads();

    // Remap thread indices so we write a transposed tile to A_T
    unsigned int out_col = blockIdx.y * TILE + threadIdx.x;
    unsigned int out_row = blockIdx.x * TILE + threadIdx.y;

    // Write tile to A_T: threads in a warp write consecutive columns
    // (which are consecutive rows of A_T) → coalesced global memory write
    if (out_row < n && out_col < n)
        A_T[out_row * n + out_col] = tile[threadIdx.x][threadIdx.y];
}

// =============================================================================
// Dataset helpers
// =============================================================================
cuDoubleComplex func_Gxy(double x, double y, size_t fx, size_t fy)
{
    return make_cuDoubleComplex(
        cos(2.0 * M_PI * fx * x) * cos(2.0 * M_PI * fy * y), 0.0);
}

struct dataset { double *grid_x, *grid_y; cuDoubleComplex *Gxy; };

struct dataset create_dataset(size_t n, size_t fx, size_t fy)
{
    struct dataset d;
    d.grid_x = (double*)malloc(n * sizeof(double));
    d.grid_y = (double*)malloc(n * sizeof(double));
    d.Gxy    = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));
    for (int i = 0; i < (int)n; i++) {
        d.grid_x[i] = (double)i / n;
        d.grid_y[i] = (double)i / n;
    }
    for (int i = 0; i < (int)n; i++)
        for (int j = 0; j < (int)n; j++)
            d.Gxy[i*n+j] = func_Gxy(d.grid_x[i], d.grid_y[j], fx, fy);
    return d;
}

// =============================================================================
// Validation (same as before — result in transposed layout)
// =============================================================================
void validate_fft(const cuDoubleComplex *h, size_t n, size_t fx, size_t fy)
{
    const double tol = 1e-3;
    int errors = 0;
    size_t pi = fy, pj = fx;   // swapped: result is in transposed layout
    for (int i = 0; i < (int)n; i++)
        for (int j = 0; j < (int)n; j++) {
            cuDoubleComplex val = h[i * n + j];
            double diff;
            bool is_peak = (i == (int)pi       && j == (int)pj)
                        || (i == (int)pi       && j == (int)(n - pj))
                        || (i == (int)(n - pi) && j == (int)pj)
                        || (i == (int)(n - pi) && j == (int)(n - pj));
            diff = is_peak ? fabs(cuCreal(val) - (double)(n*n) / 4.0)
                           : cuCabs(val);
            if (diff > tol) {
                if (errors < 5)  // print first 5 only
                    printf("  Error at (%d,%d): re=%.6e im=%.6e\n",
                           i, j, cuCreal(val), cuCimag(val));
                errors++;
            }
        }
    if (errors != 0){
         printf("Validation FAILED: %d error(s).\n", errors);
    }
               
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char *argv[])
{
    if (argc < 2) {
        // fprintf(stderr, "Usage: %s <threads_per_block>\n"
        //                 "  TPB must be >= %d (= N/2) for N=%d\n"
        //                 "  Recommended: 256 or 512\n",
        //         argv[0], N/2, N);
        return 1;
    }

    int tpb = atoi(argv[1]);
    if (tpb < N / 2) {
        // Still works via the stride loop in kernel_fft_row_shared,
        // but warn the user.
        // printf("Warning: TPB=%d < N/2=%d. "
        //        "Multiple iterations per thread in butterfly loop.\n\n",
        //        tpb, N/2);
    }
    if (tpb > 1024) {
        // fprintf(stderr, "Error: TPB=%d exceeds CUDA max (1024).\n", tpb);
        return 1;
    }

    const unsigned int n      = N;
    const unsigned int n_bits = (unsigned int)log2((double)n);
    const size_t       size   = n * n * sizeof(cuDoubleComplex);
    const size_t       fx = 2, fy = 2;

    // Shared memory per block for the FFT kernel
    const size_t smem_size = n * sizeof(cuDoubleComplex);   // 8192 bytes

    // printf("2D FFT — shared memory, FP64, N=%d, TPB=%d\n", n, tpb);
    // printf("Shared memory per block: %zu bytes\n\n", smem_size);

    // -------------------------------------------------------------------------
    // Host allocation and dataset
    // -------------------------------------------------------------------------
    struct dataset   data     = create_dataset(n, fx, fy);
    cuDoubleComplex *h_result = (cuDoubleComplex*)malloc(size);

    // -------------------------------------------------------------------------
    // Device allocation
    // -------------------------------------------------------------------------
    cuDoubleComplex *d_A, *d_A_T;
    cudaMalloc(&d_A,   size);
    cudaMalloc(&d_A_T, size);

    // -------------------------------------------------------------------------
    // Timing events
    // -------------------------------------------------------------------------
    cudaEvent_t ev_start, ev_stop, HtD_start, HtD_stop, DtH_start, DtH_stop;
    cudaEventCreate(&ev_start);  cudaEventCreate(&ev_stop);
    cudaEventCreate(&HtD_start); cudaEventCreate(&HtD_stop);
    cudaEventCreate(&DtH_start); cudaEventCreate(&DtH_stop);

    // -------------------------------------------------------------------------
    // Host → Device
    // -------------------------------------------------------------------------
    cudaEventRecord(HtD_start);
    cudaMemcpy(d_A, data.Gxy, size, cudaMemcpyHostToDevice);
    cudaEventRecord(HtD_stop);
    cudaEventSynchronize(HtD_stop);
    float ms_HtD; cudaEventElapsedTime(&ms_HtD, HtD_start, HtD_stop);

    // -------------------------------------------------------------------------
    // Grid / block configs
    //
    // FFT kernel: one block per row, TPB threads per block
    //   grid = (1, N) — blockIdx.y is the row index
    //
    // Transpose kernel: TILE×TILE blocks over the full N×N matrix
    // -------------------------------------------------------------------------
    dim3 fft_grid(1, n);
    dim3 fft_block(tpb);

    dim3 tr_grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    dim3 tr_block(TILE, TILE);

    // -------------------------------------------------------------------------
    // Warmup (avoids cold-start timing artefacts)
    // -------------------------------------------------------------------------
    kernel_fft_row_shared<<<fft_grid, fft_block, smem_size>>>(d_A, n, n_bits);
    cudaDeviceSynchronize();
    // Restore input after warmup
    cudaMemcpy(d_A, data.Gxy, size, cudaMemcpyHostToDevice);

    // -------------------------------------------------------------------------
    // Computation
    // -------------------------------------------------------------------------
    cudaEventRecord(ev_start);

    // STEP 1: FFT every row (shared memory, all stages in one kernel launch)
    kernel_fft_row_shared<<<fft_grid, fft_block, smem_size>>>(d_A, n, n_bits);
    cudaDeviceSynchronize();

    // STEP 2: Tiled transpose (coalesced read + write)
    kernel_transpose_tiled<<<tr_grid, tr_block>>>(d_A, d_A_T, n);
    cudaDeviceSynchronize();

    // STEP 3: FFT every row of transposed matrix (= column FFTs of original)
    kernel_fft_row_shared<<<fft_grid, fft_block, smem_size>>>(d_A_T, n, n_bits);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ms_compute; cudaEventElapsedTime(&ms_compute, ev_start, ev_stop);

    // -------------------------------------------------------------------------
    // Device → Host
    // -------------------------------------------------------------------------
    cudaEventRecord(DtH_start);
    cudaMemcpy(h_result, d_A_T, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(DtH_stop);
    cudaEventSynchronize(DtH_stop);
    float ms_DtH; cudaEventElapsedTime(&ms_DtH, DtH_start, DtH_stop);

    // -------------------------------------------------------------------------
    // Report
    // -------------------------------------------------------------------------
    // Kernel counts (for bandwidth comparison with global version):
    //   global version: 2*(1 bit-rev + 9 butterfly) + 1 transpose = 21 launches
    //   shared version: 2*(1 fft_row)               + 1 transpose =  3 launches
    double flops   = 21.0 * (double)n * (double)n * log2((double)n);
    // Shared version global memory traffic:
    //   fft_row (×2):   2 passes × N rows × (N reads + N writes) = 4*N^2*16 bytes
    //   transpose (×1): N^2 reads + N^2 writes                   = 2*N^2*16 bytes
    double bytes   = (4.0 + 2.0) * (double)n * (double)n * sizeof(cuDoubleComplex);
    double sec     = ms_compute * 1e-3;

    printf("%.9f\n", sec);

    // printf("Kernel launches    : 3  (vs 21 in global-memory version)\n");
    // printf("Compute time       : %.6f ms\n", ms_compute);
    // printf("HtD transfer       : %.4f ms\n", ms_HtD);
    // printf("DtH transfer       : %.4f ms\n", ms_DtH);
    // printf("\n");
    // printf("GFLOPS             : %.3f  (T4 FP64 peak: 254 GFLOPS)\n",
    //        flops / sec / 1e9);
    // printf("Effective BW       : %.2f GB/s  (T4 peak: 320 GB/s)\n",
    //        bytes / sec / 1e9);
    // printf("BW utilization     : %.1f%%\n",
    //        100.0 * (bytes / sec / 1e9) / 320.0);
    // printf("Arith. intensity   : %.3f FLOP/byte  (ridge: 0.79)\n",
    //        flops / bytes);
    // printf("\n");

    validate_fft(h_result, n, fx, fy);

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    cudaFree(d_A); cudaFree(d_A_T);
    cudaEventDestroy(ev_start);  cudaEventDestroy(ev_stop);
    cudaEventDestroy(HtD_start); cudaEventDestroy(HtD_stop);
    cudaEventDestroy(DtH_start); cudaEventDestroy(DtH_stop);
    free(h_result);
    free(data.grid_x); free(data.grid_y); free(data.Gxy);

    return 0;
}
