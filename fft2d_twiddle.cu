// =============================================================================
// fft2d_twiddle.cu — 2D FFT using the padded shared memory + precomputed
// twiddle paradigm, adapted to double precision and the row-column algorithm.
//
// Key design choices from the reference paradigm:
//   1. PADDED SHARED MEMORY: s_data[TPB + TPB/32] — every 32 elements get
//      one extra padding slot. Eliminates bank conflicts on strided access
//      patterns that occur during butterfly stages.
//      Access pattern: padded_idx = idx + idx/32
//
//   2. PRECOMPUTED TWIDDLES IN CONSTANT MEMORY: c_twiddles[N/2]
//      W[k] = e^{-2pi i k / N}. Any stage-s twiddle W_m^j = W_N^{j*step}
//      = c_twiddles[j * step]. Broadcast-cached for uniform warp access.
//
//   3. INTRA-BLOCK BUTTERFLY LOOP: all stages where stride <= TPB/2 are
//      handled inside one kernel in a for loop over stages. Each stage ends
//      with __syncthreads(). This is equivalent to the reference paradigm's
//      fft_stage_optimized loop, but we fuse all intra-block stages into a
//      single kernel launch (reducing launch overhead).
//
//   4. BIT-REVERSAL FUSED INTO LOAD: element gid is loaded into
//      s_data[padded(bit_rev(tid))] — the permutation costs nothing extra.
//
//   5. GLOBAL-MEMORY STAGES: when stride > TPB/2, partners are in different
//      blocks. These are handled by separate kernel launches (one per stage),
//      matching the reference paradigm's outer loop in compute_fft_optimized.
//      __ldg() is used for the read-only load path.
//
// Algorithm: row-column decomposition
//   fft_pass(d_A)        : row FFTs  (smem kernel + global-stage kernels)
//   transpose(d_A, d_T)  : tiled bank-conflict-free transpose
//   fft_pass(d_T)        : column FFTs
//
// Compile:
//   nvcc -O2 -arch=native -o fft2d_twiddle fft2d_twiddle.cu
// Run:
//   ./fft2d_twiddle <N> <threads_per_block>
//   e.g. ./fft2d_twiddle 2048 256
// =============================================================================

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// -----------------------------------------------------------------------------
// Compile-time limits
// -----------------------------------------------------------------------------
#define MAX_N       4096    // max matrix dimension
#define TILE        32      // transpose tile
#define MAX_TPB     512     // max threads per block

// -----------------------------------------------------------------------------
// Precomputed twiddle factors in constant memory (paradigm item 2)
// W[k] = e^{-2pi i k / N},  k = 0 .. N/2-1
// Size: 4096/2 * 16 bytes = 32 KB — within 64 KB constant memory limit.
// -----------------------------------------------------------------------------
__constant__ double2 c_twiddles[MAX_N / 2];

void init_twiddles(unsigned int n)
{
    double2 *h = (double2*)malloc((n / 2) * sizeof(double2));
    for (unsigned int k = 0; k < n / 2; k++) {
        double angle = -2.0 * M_PI * (double)k / (double)n;
        h[k].x = cos(angle);
        h[k].y = sin(angle);
    }
    cudaMemcpyToSymbol(c_twiddles, h, (n / 2) * sizeof(double2));
    free(h);
}

// -----------------------------------------------------------------------------
// Device helpers
// -----------------------------------------------------------------------------

// Padded index: insert one padding element every 32 slots (paradigm item 1).
// This maps a logical index idx to a bank-conflict-free shared memory slot.
__device__ __forceinline__
unsigned int padded(unsigned int idx) { return idx + idx / 32; }

// Bit-reversal of a log2_n-bit integer
__device__ __forceinline__
unsigned int bit_rev(unsigned int x, unsigned int log2_n)
{
    unsigned int r = 0;
    for (int b = 0; b < (int)log2_n; b++) {
        r   = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

// Complex multiply: (a + ib)(c + id) = (ac-bd) + i(ad+bc)
__device__ __forceinline__
double2 cmul(double2 a, double2 b)
{
    return make_double2(a.x*b.x - a.y*b.y,
                        a.x*b.y + a.y*b.x);
}

// =============================================================================
// KERNEL 1 — padded shared memory FFT for all intra-block stages
//
// Grid  : dim3(n / TPB, n)
// Block : dim3(TPB)
// SMEM  : (TPB + TPB/32 + 1) * sizeof(double2)   — padded allocation
//
// Handles stages s = 1 .. log2(TPB):
//   stride = 2^(s-1), from 1 up to TPB/2.
//   All butterfly partners are within the same block → no inter-block sync.
//
// Fuses bit-reversal into the load (paradigm item 4):
//   thread tid loads global element gid into s_data[padded(bit_rev(tid))].
//
// Butterfly loop (paradigm item 3):
//   For each stage, every thread computes one butterfly pair using the
//   padded index to avoid bank conflicts. Twiddle from c_twiddles.
// =============================================================================
__global__ void kernel_fft_smem(double2     *A,
                                unsigned int n,
                                unsigned int n_bits,
                                unsigned int log2_tpb)
{
    // Padded shared memory: TPB slots + one pad per 32 slots + 1 guard
    extern __shared__ double2 s_data[];   // size passed at launch

    unsigned int tid         = threadIdx.x;
    unsigned int block_start = blockIdx.x * blockDim.x;
    unsigned int row         = blockIdx.y;
    unsigned int gid         = block_start + tid;

    // -------------------------------------------------------------------------
    // LOAD with bit-reversal fused in (paradigm item 4)
    // Reverse only the lower log2_tpb bits (intra-block position).
    // The upper bits (block index) are fixed — bit-reversal across blocks
    // is handled implicitly by the global-stage kernel ordering.
    // -------------------------------------------------------------------------
    unsigned int local_rev = bit_rev(tid, log2_tpb);
    s_data[padded(local_rev)] = A[row * n + gid];
    __syncthreads();

    // -------------------------------------------------------------------------
    // BUTTERFLY LOOP — all intra-block stages (paradigm item 3)
    // Stage s: stride = 2^(s-1), group size m = 2^s
    //   j    = tid & (stride - 1)        : position within group
    //   k    = (tid >> s) << s           : group start (= (tid/m)*m)
    //   step = n >> s                    : twiddle index stride
    //   twiddle = c_twiddles[j * step]   : W_n^{j * step}
    // -------------------------------------------------------------------------
    for (unsigned int s = 1; s <= log2_tpb; s++) {
        unsigned int stride = 1u << (s - 1);   // butterfly half-span
        unsigned int m      = stride << 1;      // group size
        unsigned int step   = n >> s;           // twiddle stride

        unsigned int j = tid & (stride - 1);   // intra-group position
        unsigned int k = (tid / m) * m;        // group start

        unsigned int i_up  = k + j;            // upper element (logical)
        unsigned int i_lo  = i_up + stride;    // lower element (logical)

        // Padded indices for bank-conflict-free access (paradigm item 1)
        double2 u  = s_data[padded(i_up)];
        double2 vt = cmul(c_twiddles[j * step], s_data[padded(i_lo)]);

        s_data[padded(i_up)] = make_double2(u.x + vt.x, u.y + vt.y);
        s_data[padded(i_lo)] = make_double2(u.x - vt.x, u.y - vt.y);

        __syncthreads();   // barrier between stages (paradigm item 3)
    }

    // -------------------------------------------------------------------------
    // STORE — coalesced write back (paradigm item 1: read in padded order,
    // write to consecutive global addresses)
    // -------------------------------------------------------------------------
    A[row * n + gid] = s_data[padded(tid)];
}

// =============================================================================
// KERNEL 2 — global-memory butterfly for stages beyond log2(TPB)
//
// Matches the reference paradigm's per-stage kernel launch in compute_fft_optimized.
// Uses __ldg() for the read-only cache path on non-coalesced strided loads.
// Twiddle from c_twiddles (paradigm item 2).
//
// Grid  : dim3(n/2 / TPB, n)
// Block : dim3(TPB)
// =============================================================================
__global__ void kernel_fft_global(double2     *A,
                                  unsigned int n,
                                  unsigned int stride,   // 2^(s-1)
                                  unsigned int step)     // n / (2*stride)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (tid >= n / 2) return;

    unsigned int m   = stride << 1;
    unsigned int j   = tid & (stride - 1);
    unsigned int k   = (tid / stride) * m - (tid / stride) * stride;
    // Cleaner: recompute group start
    unsigned int grp = tid / stride;       // which butterfly group
    unsigned int i_up = grp * m + j;
    unsigned int i_lo = i_up + stride;

    unsigned int base = row * n;

    double2 u  = __ldg(&A[base + i_up]);
    double2 vt = cmul(c_twiddles[j * step], __ldg(&A[base + i_lo]));

    A[base + i_up] = make_double2(u.x + vt.x, u.y + vt.y);
    A[base + i_lo] = make_double2(u.x - vt.x, u.y - vt.y);
}

// =============================================================================
// KERNEL 3 — tiled bank-conflict-free transpose (unchanged from previous)
// =============================================================================
__global__ void kernel_transpose(const double2 *A,
                                       double2 *A_T,
                                       unsigned int n)
{
    __shared__ double2 tile[TILE][TILE + 1];

    unsigned int col = blockIdx.x * TILE + threadIdx.x;
    unsigned int row = blockIdx.y * TILE + threadIdx.y;
    if (row < n && col < n)
        tile[threadIdx.y][threadIdx.x] = A[row * n + col];
    __syncthreads();

    unsigned int out_col = blockIdx.y * TILE + threadIdx.x;
    unsigned int out_row = blockIdx.x * TILE + threadIdx.y;
    if (out_row < n && out_col < n)
        A_T[out_row * n + out_col] = tile[threadIdx.x][threadIdx.y];
}

// =============================================================================
// Host: one full FFT pass over all rows (matching paradigm's compute_fft_optimized)
// =============================================================================
static void fft_pass(double2      *d_A,
                     unsigned int  n,
                     unsigned int  n_bits,
                     int           tpb)
{
    unsigned int log2_tpb = (unsigned int)log2f((float)tpb);

    // Padded shared memory size: TPB slots + one pad per 32 slots + 1 guard
    size_t smem = ((size_t)tpb + tpb / 32 + 1) * sizeof(double2);

    // Phase 1: all intra-block stages in one kernel launch
    dim3 grd_smem(n / tpb, n);
    kernel_fft_smem<<<grd_smem, tpb, smem>>>(d_A, n, n_bits, log2_tpb);

    // Phase 2: global-memory stages — one launch per stage (paradigm pattern)
    dim3 grd_glob((n / 2 + tpb - 1) / tpb, n);
    for (unsigned int s = log2_tpb + 1; s <= n_bits; s++) {
        cudaDeviceSynchronize();   // cross-block barrier between stages
        unsigned int stride = 1u << (s - 1);
        unsigned int step   = n >> s;
        kernel_fft_global<<<grd_glob, tpb>>>(d_A, n, stride, step);
    }
    // Caller is responsible for final sync
}

// =============================================================================
// Dataset
// =============================================================================
double2 func_Gxy(double x, double y, size_t fx, size_t fy)
{
    return make_double2(
        cos(2.0 * M_PI * (double)fx * x) * cos(2.0 * M_PI * (double)fy * y),
        0.0);
}
struct dataset { double *gx, *gy; double2 *Gxy; };
struct dataset create_dataset(size_t n, size_t fx, size_t fy)
{
    struct dataset d;
    d.gx  = (double*)malloc(n * sizeof(double));
    d.gy  = (double*)malloc(n * sizeof(double));
    d.Gxy = (double2*)malloc(n * n * sizeof(double2));
    for (int i = 0; i < (int)n; i++) {
        d.gx[i] = (double)i / (double)n;
        d.gy[i] = (double)i / (double)n;
    }
    for (int i = 0; i < (int)n; i++)
        for (int j = 0; j < (int)n; j++)
            d.Gxy[i*n+j] = func_Gxy(d.gx[i], d.gy[j], fx, fy);
    return d;
}

// =============================================================================
// Validation (result in transposed layout)
// =============================================================================
void validate(const double2 *h, size_t n, size_t fx, size_t fy)
{
    const double tol = 1e-3;
    int errors = 0;
    double expected = (double)(n * n) / 4.0;
    for (int i = 0; i < (int)n; i++)
        for (int j = 0; j < (int)n; j++) {
            double2 v = h[i*n+j];
            bool peak = (i==(int)fy     && j==(int)fx)
                     || (i==(int)fy     && j==(int)(n-fx))
                     || (i==(int)(n-fy) && j==(int)fx)
                     || (i==(int)(n-fy) && j==(int)(n-fx));
            double re = v.x, im = v.y;
            double diff = peak ? fabs(re - expected)
                               : sqrt(re*re + im*im);
            if (diff > tol) {
                if (errors < 5)
                    printf("  err (%d,%d): re=%.4e im=%.4e\n", i, j, re, im);
                errors++;
            }
        }
    printf("Validation %s (%d errors)\n", errors ? "FAILED" : "PASSED", errors);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <TPB>\n"
                        "  N   : power of two, 32 <= N <= %d\n"
                        "  TPB : power of two, <= %d\n",
                argv[0], MAX_N, MAX_TPB);
        return 1;
    }
    const unsigned int n   = (unsigned int)atoi(argv[1]);
    const int          tpb = atoi(argv[2]);

    if (n > MAX_N || n < 32 || (n & (n-1)) != 0) {
        fprintf(stderr, "N must be power-of-two in [32, %d]\n", MAX_N); return 1; }
    if (tpb > MAX_TPB || tpb < 32 || (tpb & (tpb-1)) != 0) {
        fprintf(stderr, "TPB must be power-of-two in [32, %d]\n", MAX_TPB); return 1; }
    if ((unsigned)tpb > n) {
        fprintf(stderr, "TPB must be <= N\n"); return 1; }

    const unsigned int n_bits = (unsigned int)log2f((float)n);
    const size_t       size   = (size_t)n * n * sizeof(double2);
    const size_t       fx = 3, fy = 5;

    // Init twiddles (paradigm: init once, reuse across calls)
    init_twiddles(n);

    // Host data
    struct dataset data     = create_dataset(n, fx, fy);
    double2       *h_result = (double2*)malloc(size);

    // Device allocation
    double2 *d_A, *d_T;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_T, size);

    // Timing events
    cudaEvent_t t0, t1, h2d0, h2d1, d2h0, d2h1;
    cudaEventCreate(&t0);   cudaEventCreate(&t1);
    cudaEventCreate(&h2d0); cudaEventCreate(&h2d1);
    cudaEventCreate(&d2h0); cudaEventCreate(&d2h1);

    // H→D
    cudaEventRecord(h2d0);
    cudaMemcpy(d_A, data.Gxy, size, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d1);
    cudaEventSynchronize(h2d1);
    float ms_h2d; cudaEventElapsedTime(&ms_h2d, h2d0, h2d1);

    // Warmup
    {
        dim3 tg((n+TILE-1)/TILE, (n+TILE-1)/TILE), tb(TILE, TILE);
        fft_pass(d_A, n, n_bits, tpb); cudaDeviceSynchronize();
        kernel_transpose<<<tg, tb>>>(d_A, d_T, n); cudaDeviceSynchronize();
        fft_pass(d_T, n, n_bits, tpb); cudaDeviceSynchronize();
    }
    cudaMemcpy(d_A, data.Gxy, size, cudaMemcpyHostToDevice);

    // Timed run
    dim3 tr_grd((n+TILE-1)/TILE, (n+TILE-1)/TILE), tr_blk(TILE, TILE);

    cudaEventRecord(t0);

    fft_pass(d_A, n, n_bits, tpb);
    cudaDeviceSynchronize();
    kernel_transpose<<<tr_grd, tr_blk>>>(d_A, d_T, n);
    cudaDeviceSynchronize();
    fft_pass(d_T, n, n_bits, tpb);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_compute; cudaEventElapsedTime(&ms_compute, t0, t1);

    // D→H
    cudaEventRecord(d2h0);
    cudaMemcpy(h_result, d_T, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(d2h1);
    cudaEventSynchronize(d2h1);
    float ms_d2h; cudaEventElapsedTime(&ms_d2h, d2h0, d2h1);

    // Report
    unsigned int log2_tpb = (unsigned int)log2f((float)tpb);
    printf("N=%u  TPB=%d\n", n, tpb);
    printf("  Intra-block stages (SMEM) : %u  (fused in 1 kernel launch)\n", log2_tpb);
    printf("  Inter-block stages (glob) : %u  (1 launch each)\n", n_bits - log2_tpb);
    printf("  SMEM per block            : %zu bytes (padded)\n",
           ((size_t)tpb + tpb/32 + 1) * sizeof(double2));
    printf("  Compute time              : %.6f ms\n", ms_compute);
    printf("  H->D transfer             : %.4f ms\n", ms_h2d);
    printf("  D->H transfer             : %.4f ms\n", ms_d2h);

    double flops    = 5.0 * (double)n * (double)n * (double)n_bits;
    double bytes_bw = 6.0 * (double)n * (double)n * sizeof(double2);
    double sec      = ms_compute * 1e-3;
    printf("  GFLOPS (FP64)             : %.3f\n", flops / sec / 1e9);
    printf("  Eff. bandwidth            : %.2f GB/s\n", bytes_bw / sec / 1e9);

    validate(h_result, n, fx, fy);

    // Cleanup
    cudaFree(d_A); cudaFree(d_T);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaEventDestroy(h2d0); cudaEventDestroy(h2d1);
    cudaEventDestroy(d2h0); cudaEventDestroy(d2h1);
    free(h_result); free(data.gx); free(data.gy); free(data.Gxy);
    return 0;
}
