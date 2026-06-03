// define dataset function (2D)
cuDoubleComplex func_Gxy(const double x, const double y, size_t fx, size_t fy){
    return make_cuDoubleComplex(cos(2 * M_PI * fx * x) * cos(2 * M_PI * fy * y), 0.0);
}

struct dataset{
    double *grid_x;
    double *grid_y;
    cuDoubleComplex *Gxy;
};

struct dataset create_dataset_cuda(size_t n, size_t fx, size_t fy){
    struct dataset d;
    d.grid_x = (double*)malloc(n * sizeof(double));
    d.grid_y = (double*)malloc(n * sizeof(double));
    d.Gxy = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));
    for(int i=0; i < n; i++){
        d.grid_x[i] = (double)i / (double)n;
        d.grid_y[i] = (double)i / (double)n;
    }
    for(int i=0; i < n; i++)
        for(int j=0; j < n; j++)
            d.Gxy[i * n + j] = func_Gxy(d.grid_x[i], d.grid_y[j], fx, fy);
    return d;
}

/* ------------------------------------------------------------------ */
/* Twiddle kernel                                                      */
/* ------------------------------------------------------------------ */
__global__ void kernel_precompute_twiddles(cuDoubleComplex *tw, unsigned int n)
{
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n / 2) {
        double s, c;
        sincos(-2.0 * M_PI * (double)k / (double)n, &s, &c);
        tw[k] = make_cuDoubleComplex(c, s);
    }
}

/* ------------------------------------------------------------------ */
/* Shared memory FFT kernel                                           */
/*                                                                    */
/* One block per row. The full row lives in shared memory for all    */
/* butterfly stages — only 1 global load and 1 global store per row. */
/*                                                                    */
/* Launch: grid(N, 1), block(TPB, 1), smem = N*sizeof(complex)       */
/* ------------------------------------------------------------------ */
__global__ void kernel_fft_smem(cuDoubleComplex       *d_mat,
                                 const cuDoubleComplex *d_tw,
                                 unsigned int           n,
                                 unsigned int           n_bits)
{
    extern __shared__ cuDoubleComplex smem[];

    unsigned int row        = blockIdx.x;
    unsigned int tid        = threadIdx.x;
    unsigned int row_offset = row * n;

    /* load row into smem with bit-reversal applied during load */
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        unsigned int r = 0, tmp = i;
        for (unsigned int b = 0; b < n_bits; b++) {
            r   = (r << 1) | (tmp & 1);
            tmp >>= 1;
        }
        smem[r] = d_mat[row_offset + i];
    }
    __syncthreads();

    /* all butterfly stages in shared memory */
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m    = 1u << s;
        unsigned int half = m >> 1;
        unsigned int step = n / m;

        for (unsigned int tid_bf = tid; tid_bf < n / 2; tid_bf += blockDim.x) {
            unsigned int k = (tid_bf / half) * m;
            unsigned int j =  tid_bf % half;
            cuDoubleComplex w = d_tw[j * step];
            cuDoubleComplex t = cuCmul(w, smem[k + j + half]);
            cuDoubleComplex u = smem[k + j];
            smem[k + j]        = cuCadd(u, t);
            smem[k + j + half] = cuCsub(u, t);
        }
        __syncthreads();
    }

    /* write result back to global memory */
    for (unsigned int i = tid; i < n; i += blockDim.x)
        d_mat[row_offset + i] = smem[i];
}

/* ------------------------------------------------------------------ */
/* Tiled transpose with shared memory — avoids uncoalesced writes    */
/* +1 padding eliminates bank conflicts                               */
/* ------------------------------------------------------------------ */
#define TILE_DIM 16

__global__ void matrixTransposition(cuDoubleComplex *A,
                                     cuDoubleComplex *A_T,
                                     int width)
{
    __shared__ cuDoubleComplex tile[TILE_DIM][TILE_DIM + 1];

    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    if (row < width && col < width)
        tile[threadIdx.y][threadIdx.x] = A[row * width + col];
    __syncthreads();

    col = blockIdx.y * TILE_DIM + threadIdx.x;
    row = blockIdx.x * TILE_DIM + threadIdx.y;

    if (row < width && col < width)
        A_T[row * width + col] = tile[threadIdx.x][threadIdx.y];
}

/* ------------------------------------------------------------------ */
/* Validation                                                         */
/* ------------------------------------------------------------------ */
void validate_fft(cuDoubleComplex *h_X_kl, size_t n, size_t fx, size_t fy)
{
    double tolerance = 1e-3;
    int n_errors = 0;
    for (int i = 0; i < (int)n; i++) {
        for (int j = 0; j < (int)n; j++) {
            cuDoubleComplex val = h_X_kl[i * n + j];
            double diff = 0.0;
            if ((i == fx && j == fy) || (i == fx && j == n - fy) ||
                (i == n - fx && j == fy) || (i == n - fx && j == n - fy))
                diff = fabs(cuCreal(val) - pow(n, 2) / 4.0);
            else
                diff = cuCabs(val);
            if (diff > tolerance)
                printf("Error at (%d,%d): (%e,%e)\n", i, j, cuCreal(val), cuCimag(val));
        }
    }
}