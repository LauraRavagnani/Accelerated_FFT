// =============================================================================
// 3 kernels:
//   1. kernel_bit_reverse   : in-place bit-reversal, all rows, 2D grid
//   2. kernel_butterfly     : all stages × all rows, 2D grid, one launch per stage
//   3. kernel_transpose     : out-of-place matrix transpose
// =============================================================================


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
// Grid  : dim3(n/TPB,  n)   x=position in row, y=row index
// Block : dim3(TPB)
//
// Each thread reverses the bits of its column index `tid`
// -----------------------------------------------------------------------------
__global__ void kernel_bit_reverse(cuDoubleComplex *A,
                                   unsigned int     n,
                                   unsigned int     n_bits)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (tid >= n) return;    // need only 512 threads (tid from 0 to 511)

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
// Grid  : dim3((n/2)/TPB,  n)   x=butterfly index, y=row index
// Block : dim3(TPB)
//
// Called once per stage s = 1, …, log2(n)
//
// Each thread computes one butterfly pair (u, t) in global memory
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

    unsigned int k = (tid / half) * m;   // start of the group
    unsigned int j =  tid % half;        // position within the group

    // twiddle factor  W_n^{j * step} = e^{-2πi * j * step / n}
    double angle = -2.0 * M_PI * (double)(j * step) / (double)n;
    double c, s;
    sincos(angle, &s, &c);
    cuDoubleComplex w = make_cuDoubleComplex(c, s);

    cuDoubleComplex u = A[row * n + k + j];
    cuDoubleComplex t = cuCmul(w, A[row * n + k + j + half]);

    A[row * n + k + j]        = cuCadd(u, t);
    A[row * n + k + j + half] = cuCsub(u, t);       
}

// -----------------------------------------------------------------------------
// KERNEL 3 — out-of-place matrix transpose
//
// Grid  : dim3(n/TILE,  n/TILE)
// Block : dim3(TILE, TILE)
//
// Reads A[row][col] and writes A_T[col][row].
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
// Validation
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
