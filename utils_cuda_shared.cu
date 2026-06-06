// =============================================================================
// utils_cuda_shared.cu
// 2D FFT using shared memory — exactly 2 kernels:
//   1. kernel_fft_row_smem  : bit-reverse + all butterfly stages in smem,
//                             one block per row, one launch per FFT pass
//   2. kernel_transpose     : out-of-place transpose using a shared memory
//                             tile to coalesce both reads and writes
//
// Key difference from the global memory version:
//   - All FFT arithmetic stays in the 48 KB on-chip scratchpad
//   - DRAM is touched exactly once per row (load) + once (store)
//   - No cudaDeviceSynchronize() between butterfly stages — __syncthreads()
//     is sufficient because one block owns the entire row
//   - The transpose also uses smem to fix the uncoalesced write that the
//     naive global-memory transpose has
// =============================================================================

#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// -----------------------------------------------------------------------------
// Dataset helpers  (identical to global version)
// -----------------------------------------------------------------------------

cuDoubleComplex func_Gxy(const double x, const double y,
                         size_t fx, size_t fy)
{
    return make_cuDoubleComplex(
        cos(2.0 * M_PI * fx * x) * cos(2.0 * M_PI * fy * y), 0.0);
}

struct dataset {
    double          *grid_x;
    double          *grid_y;
    cuDoubleComplex *Gxy;
};

struct dataset create_dataset(size_t n, size_t fx, size_t fy)
{
    struct dataset d;
    d.grid_x = (double*)malloc(n * sizeof(double));
    d.grid_y = (double*)malloc(n * sizeof(double));
    d.Gxy    = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));

    for (int i = 0; i < (int)n; i++) {
        d.grid_x[i] = (double)i / (double)n;
        d.grid_y[i] = (double)i / (double)n;
    }
    for (int i = 0; i < (int)n; i++)
        for (int j = 0; j < (int)n; j++)
            d.Gxy[i * n + j] = func_Gxy(d.grid_x[i], d.grid_y[j], fx, fy);

    return d;
}

// -----------------------------------------------------------------------------
// KERNEL 1 — bit-reverse + all butterfly stages in shared memory
//
// Launch config (fixed — NOT parameterised by TPB):
//   blockDim.x = n / 2      (each thread owns exactly one butterfly pair)
//   gridDim.x  = n          (one block per row)
//   dynamic smem = n * sizeof(cuDoubleComplex)   (the full row)
//
// Why blockDim = n/2 and not n?
//   Each butterfly stage maps n/2 thread indices to n/2 (u,t) pairs.
//   Using n/2 threads means every thread does exactly one butterfly per stage
//   with no idle threads and no conditional guards inside the stage loop.
//   The two elements loaded/stored per thread (tid and tid+n/2) cover the
//   full row of n elements during the load/store phases.
//
// Memory traffic vs global version:
//   Global : 2 * (n/2) * log2(n) reads  +  2 * (n/2) * log2(n) writes  to DRAM
//   Shared : n reads from DRAM (load) + n writes to DRAM (store)
//            everything in between stays on-chip
// -----------------------------------------------------------------------------
__global__ void kernel_fft_row_smem(cuDoubleComplex *A,
                                    unsigned int     n,
                                    unsigned int     n_bits)
{
    // one block = one row; dynamic shared memory holds the entire row
    extern __shared__ cuDoubleComplex smem[];

    const unsigned int tid = threadIdx.x;   // 0 .. n/2 - 1
    const unsigned int row = blockIdx.x;    // which row this block processes
    cuDoubleComplex *row_ptr = A + row * n;

    // -------------------------------------------------------------------------
    // Phase 1 — load row from global memory into shared memory
    // Each thread loads two elements: its own index and its partner at tid+n/2
    // -------------------------------------------------------------------------
    smem[tid]         = row_ptr[tid];
    smem[tid + n / 2] = row_ptr[tid + n / 2];
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 2 — in-place bit-reversal inside shared memory
    //
    // Each thread checks index tid and tid+n/2.
    // The swap guard (r > k) ensures each pair is swapped exactly once.
    // -------------------------------------------------------------------------
    for (int base = 0; base < 2; base++) {
        unsigned int k   = tid + base * (n / 2);
        unsigned int r   = 0;
        unsigned int tmp = k;
        for (int b = 0; b < (int)n_bits; b++) {
            r   = (r << 1) | (tmp & 1);
            tmp >>= 1;
        }
        if (r > k) {
            cuDoubleComplex t = smem[k];
            smem[k]           = smem[r];
            smem[r]           = t;
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Phase 3 — all butterfly stages, entirely in shared memory
    //
    // For stage s:
    //   m    = 1 << s   : sub-DFT size
    //   half = m / 2    : number of butterfly pairs per group
    //   step = n / m    : twiddle index stride
    //
    // Thread tid maps to group k and intra-group index j:
    //   k = (tid / half) * m
    //   j =  tid % half
    //
    // Two __syncthreads() per stage:
    //   first  — all threads finish reading u and t before any thread writes
    //   second — all threads finish writing before the next stage reads
    // -------------------------------------------------------------------------
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m    = 1u << s;
        unsigned int half = m >> 1;
        unsigned int step = n / m;

        unsigned int k = (tid / half) * m;
        unsigned int j =  tid % half;

        // twiddle W_n^{j*step} = e^{-2πi * j*step / n}
        double angle = -2.0 * M_PI * (double)(j * step) / (double)n;
        double c, sv;
        sincos(angle, &sv, &c);
        cuDoubleComplex w = make_cuDoubleComplex(c, sv);

        // read from smem (on-chip, ~5 cycle latency vs ~200 for DRAM)
        cuDoubleComplex u = smem[k + j];
        cuDoubleComplex t = cuCmul(w, smem[k + j + half]);

        __syncthreads();                      // read fence

        smem[k + j]        = cuCadd(u, t);
        smem[k + j + half] = cuCsub(u, t);

        __syncthreads();                      // write fence
    }

    // -------------------------------------------------------------------------
    // Phase 4 — write result back to global memory
    // -------------------------------------------------------------------------
    row_ptr[tid]         = smem[tid];
    row_ptr[tid + n / 2] = smem[tid + n / 2];
}

// -----------------------------------------------------------------------------
// KERNEL 2 — tiled transpose using shared memory
//
// Grid  : dim3(n/TILE, n/TILE)
// Block : dim3(TILE, TILE)     where TILE=32
//
// The naive global-memory transpose (used in the global version) has
// uncoalesced *writes*: threads in a warp write to A_T[col*n + row], where
// consecutive threads have consecutive `col` but the same `row`, so writes
// are strided by n elements — one cache line per thread.
//
// The tiled fix:
//   1. Threads load a TILE×TILE block of A into smem in a coalesced pattern
//      (consecutive threads read consecutive columns → coalesced read).
//   2. __syncthreads()
//   3. Threads write to A_T reading from smem in transposed order
//      (consecutive threads write consecutive rows of A_T → coalesced write).
//
// Padding smem by 1 column (TILE+1) avoids bank conflicts on the smem read
// in step 3, where threads in a warp access the same column of smem.
// -----------------------------------------------------------------------------
static const int TILE = 32;

__global__ void kernel_transpose(const cuDoubleComplex *A,
                                       cuDoubleComplex *A_T,
                                       unsigned int     n)
{
    // +1 padding per row eliminates shared memory bank conflicts
    __shared__ cuDoubleComplex tile[TILE][TILE + 1];

    unsigned int col_in = blockIdx.x * TILE + threadIdx.x;
    unsigned int row_in = blockIdx.y * TILE + threadIdx.y;

    // coalesced read: warp reads TILE consecutive columns of one row
    if (row_in < n && col_in < n)
        tile[threadIdx.y][threadIdx.x] = A[row_in * n + col_in];

    __syncthreads();

    // transposed write coordinates
    unsigned int col_out = blockIdx.y * TILE + threadIdx.x;
    unsigned int row_out = blockIdx.x * TILE + threadIdx.y;

    // coalesced write: warp writes TILE consecutive columns of one row of A_T
    if (row_out < n && col_out < n)
        A_T[row_out * n + col_out] = tile[threadIdx.x][threadIdx.y];
}

// -----------------------------------------------------------------------------
// Validation helper  (identical to global version)
// -----------------------------------------------------------------------------
void validate_fft(const cuDoubleComplex *h, size_t n, size_t fx, size_t fy)
{
    const double tol = 1e-3;
    int errors = 0;
    for (int i = 0; i < (int)n; i++) {
        for (int j = 0; j < (int)n; j++) {
            cuDoubleComplex val = h[i * n + j];
            double diff;
            bool is_peak = (i == (int)fx       && j == (int)fy)
                        || (i == (int)fx       && j == (int)(n - fy))
                        || (i == (int)(n - fx) && j == (int)fy)
                        || (i == (int)(n - fx) && j == (int)(n - fy));
            if (is_peak)
                diff = fabs(cuCreal(val) - (double)(n * n) / 4.0);
            else
                diff = cuCabs(val);

            if (diff > tol) {
                printf("Error at (%d,%d): re=%.6e  im=%.6e\n",
                       i, j, cuCreal(val), cuCimag(val));
                errors++;
            }
        }
    }
    if (errors == 0)
        printf("Validation passed.\n");
    else
        printf("Validation FAILED: %d error(s).\n", errors);
}
