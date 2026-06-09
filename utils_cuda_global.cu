// =============================================================================
// utils_cuda_global.cu
// 2D FFT using global memory only — exactly 3 kernels:
//   1. kernel_bit_reverse   : in-place bit-reversal, all rows, 2D grid
//   2. kernel_butterfly     : all stages × all rows, 2D grid, one launch per stage
//   3. kernel_transpose     : out-of-place matrix transpose
// =============================================================================

// #include <cuComplex.h>
// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>

// -----------------------------------------------------------------------------
// Dataset helpers
// -----------------------------------------------------------------------------

cuDoubleComplex func_Gxy(const double x, const double y,
                         size_t fx, size_t fy)
{
    return make_cuDoubleComplex(
        cos(2.0 * M_PI * fx * x) * cos(2.0 * M_PI * fy * y), 0.0);
}

struct dataset {
    double         *grid_x;
    double         *grid_y;
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
// KERNEL 1 — in-place bit-reversal
//
// Grid  : dim3((n + TPB-1)/TPB,  n)   x=position in row, y=row index
// Block : dim3(TPB)
//
// Each thread reverses the bits of its column index `tid` and, if the
// reversed index `r` is strictly greater, swaps elements A[row][tid]
// and A[row][r].  The guard `r > tid` prevents double-swapping.
// -----------------------------------------------------------------------------
__global__ void kernel_bit_reverse(cuDoubleComplex *A,
                                   unsigned int     n,
                                   unsigned int     n_bits)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    //if (tid >= n) return;    // not sure it is useful

    // compute bit-reversal of tid
    unsigned int r = 0;
    unsigned int tmp = tid;
    for (int b = 0; b < (int)n_bits; b++) {
        r   = (r << 1) | (tmp & 1);
        tmp >>= 1;
    }

    if (r > tid) {           // to avoid double swapping
        cuDoubleComplex t   = A[row * n + tid];
        A[row * n + tid]    = A[row * n + r];
        A[row * n + r]      = t;
    }
}

// -----------------------------------------------------------------------------
// KERNEL 2 — Cooley-Tukey butterfly, one stage, all rows
//
// Grid  : dim3((n/2 + TPB-1)/TPB,  n)   x=butterfly index, y=row index
// Block : dim3(TPB)
//
// Called once per stage s = 1 … log2(n).
// m    = 1 << s   : current DFT sub-size
// step = n / m    : twiddle stride  (W_n^(j*step) = e^{-2πi j/m})
//
// Each thread computes ONE butterfly pair (u, t) in global memory.
// Two __syncthreads() calls bracket the read and write within the block;
// inter-block ordering is guaranteed because we call cudaDeviceSynchronize()
// in the host wrapper between stages.
// -----------------------------------------------------------------------------
__global__ void kernel_butterfly(cuDoubleComplex       *A,
                                 unsigned int           n,
                                 unsigned int           m,
                                 unsigned int           step)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row  = blockIdx.y;
    unsigned int half = m >> 1;     // since m is a power of 2, divide m by two means to remove the last 0
    if (tid >= n / 2) return;   // need only 256 threads (tid from 0 to 255)

    unsigned int k = (tid / half) * m;   // start of the current group
    unsigned int j =  tid % half;        // position within the group

    // twiddle factor  W_n^{j * step} = e^{-2πi * j * step / n}
    double angle = -2.0 * M_PI * (double)(j * step) / (double)n;
    double c, s;
    sincos(angle, &s, &c);
    cuDoubleComplex w = make_cuDoubleComplex(c, s);

    // read both operands before any write (only needed within the block)
    cuDoubleComplex u = A[row * n + k + j];
    cuDoubleComplex t = cuCmul(w, A[row * n + k + j + half]);

    A[row * n + k + j]        = cuCadd(u, t);
    A[row * n + k + j + half] = cuCsub(u, t);       
}

// -----------------------------------------------------------------------------
// KERNEL 3 — out-of-place matrix transpose
//
// Grid  : dim3((n + TILE-1)/TILE,  (n + TILE-1)/TILE)
// Block : dim3(TILE, TILE)
//
// Reads A[row][col] and writes A_T[col][row].
// Using a separate output buffer avoids the need for a tile in shared memory
// (which would be the standard coalescing trick — excluded here by design).
// -----------------------------------------------------------------------------
__global__ void kernel_transpose(const cuDoubleComplex *A,
                                       cuDoubleComplex *A_T,
                                       unsigned int     n)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n)
        A_T[col * n + row] = A[row * n + col];
}

// -----------------------------------------------------------------------------
// Validation helper
// -----------------------------------------------------------------------------
void validate_fft(const cuDoubleComplex *h, size_t n, size_t fx, size_t fy)
{
    const double tol = 1e-2;
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
    if (errors != 0)
        printf("Validation FAILED: %d error(s).\n", errors);
}
