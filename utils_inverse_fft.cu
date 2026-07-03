// =============================================================================
// 4 kernels:
//   1. kernel_bit_reverse       : in-place bit-reversal, all rows, 2D grid (same as forward)
//   2. kernel_butterfly_inverse : all stages × all rows, 2D grid, one launch per stage
//                                 (twiddle factor conjugated w.r.t. forward FFT)
//   3. kernel_transpose         : out-of-place matrix transpose (same as forward)
//   4. kernel_scale             : divide every element by n*n (1/N normalization
//                                 for the 2-D inverse transform)
// =============================================================================


// -----------------------------------------------------------------------------
// Dataset helpers
// -----------------------------------------------------------------------------

// spatial-domain reference function, used only for validation
cuDoubleComplex func_Gxy(const double x, const double y,
                         size_t fx, size_t fy)
{
    return make_cuDoubleComplex(
        cos(2.0 * M_PI * fx * x) * cos(2.0 * M_PI * fy * y), 0.0);
}

struct dataset {
    double         *grid_x;
    double         *grid_y;
    cuDoubleComplex *Gxy;      // frequency-domain input for the IFFT
};

// Builds the *frequency-domain* dataset that is the exact (unnormalized) 2-D
// DFT of  cos(2*pi*fx*x) * cos(2*pi*fy*y):  four impulses of height N*N/4 at
// (fx,fy), (fx,N-fy), (N-fx,fy), (N-fx,N-fy)  (mod N).  Feeding this through
// the IFFT below should reproduce the spatial cosine pattern, which is what
// validate_ifft checks.
struct dataset create_dataset(size_t n, size_t fx, size_t fy)
{
    struct dataset d;
    d.grid_x = (double*)malloc(n * sizeof(double));
    d.grid_y = (double*)malloc(n * sizeof(double));
    d.Gxy    = (cuDoubleComplex*)calloc(n * n, sizeof(cuDoubleComplex));

    for (int i = 0; i < (int)n; i++) {
        d.grid_x[i] = (double)i / (double)n;
        d.grid_y[i] = (double)i / (double)n;
    }

    const double peak = (double)(n * n) / 4.0;
    unsigned int fx0 = (unsigned int)fx;
    unsigned int fy0 = (unsigned int)fy;
    unsigned int fx1 = (unsigned int)(n - fx) % n;
    unsigned int fy1 = (unsigned int)(n - fy) % n;

    d.Gxy[fx0 * n + fy0] = make_cuDoubleComplex(peak, 0.0);
    d.Gxy[fx0 * n + fy1] = make_cuDoubleComplex(peak, 0.0);
    d.Gxy[fx1 * n + fy0] = make_cuDoubleComplex(peak, 0.0);
    d.Gxy[fx1 * n + fy1] = make_cuDoubleComplex(peak, 0.0);

    return d;
}

// -----------------------------------------------------------------------------
// KERNEL 1 — in-place bit-reversal  (identical to forward FFT: the
// bit-reversal permutation does not depend on transform direction)
//
// Grid  : dim3(n/TPB,  n)   x=position in row, y=row index
// Block : dim3(TPB)
// -----------------------------------------------------------------------------
__global__ void kernel_bit_reverse(cuDoubleComplex *A,
                                   unsigned int     n,
                                   unsigned int     n_bits)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (tid >= n) return;

    unsigned int r = 0;
    unsigned int tmp = tid;
    for (int b = 0; b < (int)n_bits; b++) {
        r   = (r << 1) | (tmp & 1);
        tmp >>= 1;
    }

    if (r > tid) {
        cuDoubleComplex t   = A[row * n + tid];
        A[row * n + tid]    = A[row * n + r];
        A[row * n + r]      = t;
    }
}

// -----------------------------------------------------------------------------
// KERNEL 2 — Cooley-Tukey butterfly, INVERSE, one stage, all rows
//
// Grid  : dim3((n/2)/TPB,  n)   x=butterfly index, y=row index
// Block : dim3(TPB)
//
// Called once per stage s = 1, …, log2(n)
//
// Same structure as the forward butterfly, but the twiddle factor is
// conjugated:  W_n^{-j*step} = e^{+2πi * j * step / n}
// (the 1/N scaling is applied once at the very end via kernel_scale,
//  not per-stage)
// -----------------------------------------------------------------------------
__global__ void kernel_butterfly_inverse(cuDoubleComplex       *A,
                                         unsigned int           n,
                                         unsigned int           m,
                                         unsigned int           step)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row  = blockIdx.y;
    unsigned int half = m >> 1;
    if (tid >= n / 2) return;

    unsigned int k = (tid / half) * m;
    unsigned int j =  tid % half;

    // conjugated twiddle factor: e^{+2πi * j * step / n}
    double angle = 2.0 * M_PI * (double)(j * step) / (double)n;
    double c, s;
    sincos(angle, &s, &c);
    cuDoubleComplex w = make_cuDoubleComplex(c, s);

    cuDoubleComplex u = A[row * n + k + j];
    cuDoubleComplex t = cuCmul(w, A[row * n + k + j + half]);

    A[row * n + k + j]        = cuCadd(u, t);
    A[row * n + k + j + half] = cuCsub(u, t);
}

// -----------------------------------------------------------------------------
// KERNEL 3 — out-of-place matrix transpose (identical to forward FFT)
//
// Grid  : dim3(n/TILE,  n/TILE)
// Block : dim3(TILE, TILE)
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
// KERNEL 4 — element-wise normalization  A[i] /= (n*n)
//
// A 2-D IFFT via row/column decomposition is  IDFT_rows( IDFT_cols( X ) ),
// each 1-D IDFT carrying an implicit 1/n factor. Rather than scale inside
// every butterfly stage, we fold the whole 1/(n*n) factor into one pass
// over the full matrix after both row/column transforms are done.
//
// Grid  : dim3(n/TPB, n)   x=position in row, y=row index
// Block : dim3(TPB)
// -----------------------------------------------------------------------------
__global__ void kernel_scale(cuDoubleComplex *A,
                             unsigned int     n)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (tid >= n) return;

    double inv_n2 = 1.0 / ((double)n * (double)n);
    cuDoubleComplex v = A[row * n + tid];
    A[row * n + tid] = make_cuDoubleComplex(cuCreal(v) * inv_n2,
                                            cuCimag(v) * inv_n2);
}

// -----------------------------------------------------------------------------
// Validation — compare the reconstructed spatial signal against the
// analytic cosine pattern  cos(2*pi*fx*x) * cos(2*pi*fy*y)
// -----------------------------------------------------------------------------
void validate_ifft(const cuDoubleComplex *h, size_t n, size_t fx, size_t fy)
{
    const double tol = 1e-6;
    int errors = 0;
    for (int i = 0; i < (int)n; i++) {
        double x = (double)i / (double)n;
        for (int j = 0; j < (int)n; j++) {
            double y = (double)j / (double)n;
            double expected = cos(2.0 * M_PI * fx * x) * cos(2.0 * M_PI * fy * y);

            cuDoubleComplex val = h[i * n + j];
            double diff_re = fabs(cuCreal(val) - expected);
            double diff_im = fabs(cuCimag(val));

            if (diff_re > tol || diff_im > tol) {
                printf("Error at (%d,%d): re=%.6e (expected %.6e)  im=%.6e\n",
                       i, j, cuCreal(val), expected, cuCimag(val));
                errors++;
            }
        }
    }
    if (errors != 0)
        printf("Validation FAILED: %d error(s).\n", errors);
}
