// =============================================================================
// Kernels (shared-memory optimized, tuned for Tesla T4 / sm_75):
//
//   1. kernel_fft_row_shared : FUSED bit-reversal + all Cooley-Tukey butterfly
//                              stages for one (or several) rows, entirely in
//                              shared memory. Replaces the old
//                              kernel_bit_reverse + log2(n) x kernel_butterfly
//                              global-memory launches with a SINGLE kernel
//                              launch per FFT pass:
//                                - 1 global read of the row  -> shared mem
//                                - all compute stays in shared mem (fast)
//                                - 1 global write of the row <- shared mem
//                              This removes ~2*log2(n) global memory round
//                              trips per row and ~log2(n) kernel-launch
//                              overheads.
//
//   2. kernel_bit_reverse / kernel_butterfly : kept as fallback for rows
//                              too wide to fit in one block's shared memory
//                              budget (n/2 > 1024 threads), and for the
//                              tuning path.
//
//   3. kernel_transpose_shared : classic tiled shared-memory transpose with
//                              a padded tile (TILE x TILE+1) to avoid shared
//                              memory bank conflicts, and BLOCK_ROWS-strided
//                              loads/stores so global memory accesses stay
//                              fully coalesced in both the read and the
//                              write phase.
// =============================================================================


// -----------------------------------------------------------------------------
// Dataset helpers (unchanged)
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

// =============================================================================
// FUSED shared-memory row-FFT kernel
//
// Grid  : dim3(n / rows_per_block)
// Block : dim3(n/2, rows_per_block)
// Shmem : rows_per_block * n * sizeof(cuDoubleComplex)  (dynamic)
//
// Each block owns `rows_per_block` full rows. threadIdx.y selects the row
// within the block, threadIdx.x (0..n/2-1) is the butterfly-pair index.
// The whole row lives in shared memory for the entire FFT (bit-reverse +
// all log2(n) stages); global memory is only touched once to load and
// once to store.
// =============================================================================
extern __shared__ cuDoubleComplex smem[];

__global__ void kernel_fft_row_shared(cuDoubleComplex *A,
                                      unsigned int     n,
                                      unsigned int     n_bits,
                                      unsigned int     rows_per_block)
{
    unsigned int half_n     = n >> 1;
    unsigned int local_row  = threadIdx.y;
    unsigned int row        = blockIdx.x * rows_per_block + local_row;
    unsigned int tid        = threadIdx.x;

    cuDoubleComplex *srow = smem + (size_t)local_row * n;

    // ---- coalesced load: global -> shared -------------------------------
    srow[tid]          = A[(size_t)row * n + tid];
    srow[tid + half_n] = A[(size_t)row * n + tid + half_n];
    __syncthreads();

    // ---- bit-reversal, in shared memory ----------------------------------
    // n/2 threads cover n elements: each thread handles idx = tid and
    // idx = tid + half_n (identical logic to the original single-element
    // kernel_bit_reverse, just two elements per thread).
    for (unsigned int idx = tid; idx < n; idx += half_n) {
        unsigned int r = 0, tmp = idx;
        for (unsigned int b = 0; b < n_bits; b++) {
            r   = (r << 1) | (tmp & 1u);
            tmp >>= 1;
        }
        if (r > idx) {
            cuDoubleComplex t = srow[idx];
            srow[idx] = srow[r];
            srow[r]   = t;
        }
    }
    __syncthreads();

    // ---- Cooley-Tukey butterfly stages, entirely in shared memory --------
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m    = 1u << s;
        unsigned int half = m >> 1;
        unsigned int step = n / m;

        unsigned int k = (tid / half) * m;   // start of this butterfly group
        unsigned int j =  tid % half;        // position within the group

        double angle = -2.0 * M_PI * (double)(j * step) / (double)n;
        double c, sn;
        sincos(angle, &sn, &c);
        cuDoubleComplex w = make_cuDoubleComplex(c, sn);

        cuDoubleComplex u = srow[k + j];
        cuDoubleComplex t = cuCmul(w, srow[k + j + half]);

        srow[k + j]        = cuCadd(u, t);
        srow[k + j + half] = cuCsub(u, t);

        // Different stages read/write different index groupings, so all
        // threads must finish this stage before the next one begins.
        __syncthreads();
    }

    // ---- coalesced store: shared -> global --------------------------------
    A[(size_t)row * n + tid]          = srow[tid];
    A[(size_t)row * n + tid + half_n] = srow[tid + half_n];
}

// -----------------------------------------------------------------------------
// KERNEL — in-place bit-reversal (fallback / tuning path, global memory)
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
// KERNEL — Cooley-Tukey butterfly, one stage, all rows (fallback / tuning path)
//
// Grid  : dim3((n/2)/TPB,  n)   x=butterfly index, y=row index
// Block : dim3(TPB)
// -----------------------------------------------------------------------------
__global__ void kernel_butterfly(cuDoubleComplex       *A,
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

    double angle = -2.0 * M_PI * (double)(j * step) / (double)n;
    double c, s;
    sincos(angle, &s, &c);
    cuDoubleComplex w = make_cuDoubleComplex(c, s);

    cuDoubleComplex u = A[row * n + k + j];
    cuDoubleComplex t = cuCmul(w, A[row * n + k + j + half]);

    A[row * n + k + j]        = cuCadd(u, t);
    A[row * n + k + j + half] = cuCsub(u, make_cuDoubleComplex(cuCreal(t), cuCimag(t)));
}

// =============================================================================
// KERNEL — tiled shared-memory transpose (bank-conflict free)
//
// Grid  : dim3(n/TILE_DIM, n/TILE_DIM)
// Block : dim3(TILE_DIM, BLOCK_ROWS)
//
// Classic NVIDIA transpose pattern: load a TILE_DIM x TILE_DIM tile into a
// padded shared-memory array (TILE_DIM+1 columns kills bank conflicts on
// the transposed write-back), then write it out transposed. BLOCK_ROWS 
// TILE_DIM lets each thread handle several rows of the tile so occupancy
// stays high while every global load/store remains fully coalesced.
// Bounds-checked so it also works for n < TILE_DIM.
// =============================================================================
#define TR_TILE_DIM   32
#define TR_BLOCK_ROWS 8

__global__ void kernel_transpose_shared(const cuDoubleComplex *A,
                                              cuDoubleComplex *A_T,
                                              unsigned int     n)
{
    __shared__ cuDoubleComplex tile[TR_TILE_DIM][TR_TILE_DIM + 1];

    unsigned int x = blockIdx.x * TR_TILE_DIM + threadIdx.x;
    unsigned int y = blockIdx.y * TR_TILE_DIM + threadIdx.y;

    #pragma unroll
    for (unsigned int j = 0; j < TR_TILE_DIM; j += TR_BLOCK_ROWS) {
        if (x < n && (y + j) < n)
            tile[threadIdx.y + j][threadIdx.x] = A[(size_t)(y + j) * n + x];
    }

    __syncthreads();

    x = blockIdx.y * TR_TILE_DIM + threadIdx.x;
    y = blockIdx.x * TR_TILE_DIM + threadIdx.y;

    #pragma unroll
    for (unsigned int j = 0; j < TR_TILE_DIM; j += TR_BLOCK_ROWS) {
        if (x < n && (y + j) < n)
            A_T[(size_t)(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// Kept for the tuning path / reference (naive, no shared memory)
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
// Validation (unchanged)
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