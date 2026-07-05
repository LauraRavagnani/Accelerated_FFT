/// ----------------------------------------------------------------------------- //
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


// ------------------------------------------------------------------------ //
// Dataset helpers                                                          //
// ------------------------------------------------------------------------ //
cuDoubleComplex func_Gxy(const double x, const double y, size_t fx, size_t fy){
    return make_cuDoubleComplex(cos(2.0 * M_PI * fx * x) * cos(2.0 * M_PI * fy * y), 0.0);
}

struct dataset {
    double *grid_x;
    double *grid_y;
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

// ------------------------------------------------------------------------ //
// kernel - bit-reversal + all butterfly stages                             //
//                                                                          //
// optimize the architecture as much as possible                            //                                                                         //
// Grid  : dim3(n / rows_per_block)                                         //
// Block : dim3(n/2, rows_per_block)                                        //
// Shmem : rows_per_block * n * sizeof(cuDoubleComplex)                     //
//                                                                          //
// The whole row lives in shared memory for the entire FFT                  //
// ------------------------------------------------------------------------ //
extern __shared__ cuDoubleComplex smem[];

__global__ void kernel_fft_row_shared(cuDoubleComplex *A, unsigned int n, unsigned int n_bits, unsigned int rows_per_block){
    unsigned int half_n = n >> 1;
    unsigned int local_row  = threadIdx.y;
    unsigned int row = blockIdx.x * rows_per_block + local_row;
    unsigned int tid = threadIdx.x;

    cuDoubleComplex *srow = smem + local_row * n;

    // load: global -> shared 
    srow[tid] = A[row * n + tid];
    srow[tid + half_n] = A[row * n + tid + half_n];
    __syncthreads();

    // n/2 threads cover n elements: each thread handles idx = tid and idx = tid + half_n 
    for (unsigned int idx = tid; idx < n; idx += half_n) {
        unsigned int r = 0, tmp = idx;
        for (unsigned int b = 0; b < n_bits; b++) {
            r = (r << 1) | (tmp & 1u);
            tmp >>= 1;
        }
        if (r > idx) {
            cuDoubleComplex t = srow[idx];
            srow[idx] = srow[r];
            srow[r] = t;
        }
    }
    __syncthreads();

    // butterfly stages
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m = 1u << s;
        unsigned int half = m >> 1;
        unsigned int step = n / m;

        unsigned int k = (tid / half) * m;   // start of the group
        unsigned int j =  tid % half;        // which butterfly within the group

        double angle = -2.0 * M_PI * (double)(j * step) / (double)n;
        double c, sn;
        sincos(angle, &sn, &c);
        cuDoubleComplex w = make_cuDoubleComplex(c, sn);

        cuDoubleComplex u = srow[k + j];
        cuDoubleComplex t = cuCmul(w, srow[k + j + half]);

        srow[k + j] = cuCadd(u, t);
        srow[k + j + half] = cuCsub(u, t);

        // all threads must finish this stage before the next one begins.
        __syncthreads();
    }

    // store: shared -> global
    A[(size_t)row * n + tid] = srow[tid];
    A[(size_t)row * n + tid + half_n] = srow[tid + half_n];
}

/// ----------------------------------------------------------------------- //
// kernel — in-place bit-reversal (fallback, global memory)                 //
// ------------------------------------------------------------------------ //
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

// ------------------------------------------------------------------------ //
// kernel - butterfly (fallback, global memory)                             //
// ------------------------------------------------------------------------ //
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
// kernel — tiled shared-memory transpose 
//
// Grid  : dim3(n/TILE_DIM, n/TILE_DIM)
// Block : dim3(TILE_DIM, BLOCK_ROWS)
//
// NVIDIA transpose pattern. BLOCK_ROWS TILE_DIM lets each thread handle several rows of the tile
// Bounds-checked so it also works for n < TILE_DIM.
// =============================================================================
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void kernel_transpose_shared(const cuDoubleComplex *A, cuDoubleComplex *A_T, unsigned int n){
    __shared__ cuDoubleComplex tile[TILE_DIM][TILE_DIM + 1];

    unsigned int x = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int y = blockIdx.y * TILE_DIM + threadIdx.y;

     // #pragma unroll
    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n)
            tile[threadIdx.y + j][threadIdx.x] = A[(y + j) * n + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;    // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // #pragma unroll
    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n){
            A_T[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
        } 
    }
}

// ---------------------------------------------------------------------- //
// Validation                                                             //
// four delta peaks of magnitude N^2/4 at (fx, N-fx, fy, N-fy)            //
// ---------------------------------------------------------------------- //
void validate_fft(const cuDoubleComplex *h, size_t n, size_t fx, size_t fy)
{
    const double tol = 1e-6;
    int errors = 0;
    for (int i = 0; i < (int)n; i++) {
        for (int j = 0; j < (int)n; j++) {
            cuDoubleComplex val = h[i * n + j];
            double diff = 0.0;
            if((i == fx && j == fy) || (i == fx && j == (n - fy)) || (i == (n - fx) && j == fy) || (i == (n - fx) && j == (n - fy))){
                diff = fabs(cuCreal(val) - (double)(n * n) / 4.0);
            }
            else{
                diff = cuCabs(val);
            }

            if (diff > tol) {
                printf("Error at (%d,%d): re=%.6e  im=%.6e\n", i, j, cuCreal(val), cuCimag(val));
                errors++;
            }
        }
    }
    if (errors != 0)
        printf("Validation FAILED: %d error(s).\n", errors);
}