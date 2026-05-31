#define THREADS_PER_BLOCK  2
#define N 8

// define dataset function (2D)
cuDoubleComplex func_Gxy(const double x, const double y, size_t fx, size_t fy){
	return make_cuDoubleComplex(cos(2 * M_PI * fx * x) * cos(2 * M_PI * fy * y), 0.0);
}

// define struct to return multiple objects when create dataset
struct dataset{
	double *grid_x;
	double *grid_y;
	cuDoubleComplex *Gxy;
};


// create dataset
struct dataset create_dataset_cuda(size_t n, size_t fx, size_t fy){			//return a pointer
	struct dataset d;

	d.grid_x = (double*)malloc(n * sizeof(double));
	d.grid_y = (double*)malloc(n * sizeof(double));
	d.Gxy = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));

	//create x and y grids
	for(int i=0; i < N; i++){
		d.grid_x[i] = (double)i / (double)n;
		d.grid_y[i] = (double)i / (double)n;
	}

	// create 2D dataset
	for(int i=0; i < n; i++){
		for(int j=0; j < n; j++){
			d.Gxy[i * n + j] = func_Gxy(d.grid_x[i], d.grid_y[j], fx, fy);
		}
	}

	return d;
}

/* ------------------------------------------------------------------ */
/* Bit-reversal kernel                                                */
/* ------------------------------------------------------------------ */
__global__ void kernel_bit_reverse_copy(const cuDoubleComplex *a,
                                              cuDoubleComplex *A,
                                        unsigned int n,
                                        unsigned int n_bits)
{
    auto k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        auto r   = 0;
        auto tmp = k;
        for (auto b = 0; b < n_bits; b++) {
            r   = (r << 1) | (tmp & 1);
            tmp >>= 1;
        }
        A[r] = a[k];
    }
}

/* ------------------------------------------------------------------ */
/* Twiddle kernel                                                     */
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
/* Butterfly kernel ==> compute each row independently                                                   */
/* ------------------------------------------------------------------ */
__global__ void kernel_butterfly(cuDoubleComplex *A,
                                  const cuDoubleComplex *twiddles,
                                  unsigned int n,
                                  unsigned int m,
                                  unsigned int step,
                                  unsigned int row_offset)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half = m / 2;
    if (tid < n / 2) {
        unsigned int k = (tid / half) * m;
        unsigned int j =  tid % half;
        cuDoubleComplex w = twiddles[j * step];
        cuDoubleComplex t = cuCmul(w, A[row_offset + k + j + half]);
        cuDoubleComplex u = A[row_offset + k + j];
        A[row_offset + k + j]        = cuCadd(u, t);
        A[row_offset + k + j + half] = cuCsub(u, t);
    }
}

/* ------------------------------------------------------------------ */
/* Transpose kernel     ==> 2d because transpose all matrix at once                                             */
/* ------------------------------------------------------------------ */
__global__ void matrixTransposition(cuDoubleComplex *A,
                                     cuDoubleComplex *A_T,
                                     int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        A_T[col * width + row] = A[row * width + col];
    }
}

/* ------------------------------------------------------------------ */
/* Helper: FFT on every row of a NxN matrix on device                */
/* ------------------------------------------------------------------ */
void cuda_fft(cuDoubleComplex *d_mat,
              cuDoubleComplex *d_tmp,
              cuDoubleComplex *d_tw,
              int n, int n_bits)
{
    int N_b_row = n / THREADS_PER_BLOCK;

    for (int row = 0; row < n; row++) {
        unsigned int row_offset = row * n;

        /* bit-reverse copy: read from matrix row, write to temp buffer */
        kernel_bit_reverse_copy<<<N_b_row, THREADS_PER_BLOCK>>>(d_mat + row_offset, d_tmp, n, n_bits);
        cudaDeviceSynchronize();

        /* copy bit-reversed result back into the matrix row */
        cudaMemcpy(d_mat + row_offset, d_tmp, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

        /* butterfly stages */
        for (int s = 1; s <= n_bits; s++) {
            unsigned int m    = 1u << s;
            unsigned int step = n / m;
            int N_b = (n/2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            kernel_butterfly<<<N_b, THREADS_PER_BLOCK>>>(d_mat, d_tw, n, m, step, row_offset);
            cudaDeviceSynchronize();
        }
    }
}


void validate_fft(cuDoubleComplex *h_X_kl, size_t n, size_t fx, size_t fy)
{
    double tolerance = 1e-6;
    int n_errors = 0;

    for (int i = 0; i < (int)n; i++) {
        for (int j = 0; j < (int)n; j++) {

            cuDoubleComplex val = h_X_kl[i * n + j];
            double diff = 0.0;

            if ((i == fx && j == fy)       ||
                (i == fx && j == n - fy)   ||
                (i == n - fx && j == fy)   ||
                (i == n - fx && j == n - fy)) {
                /* at peak bins: real part should equal N²/4, imag ~ 0 */
                diff = fabs(cuCreal(val) - pow(n, 2) / 4.0);
            } else {
                /* everywhere else: magnitude should be ~ 0 */
                diff = cuCabs(val);
            }

            if (diff > tolerance) {
                printf("Failed verification at (%d,%d): got (%f,%f), diff=%e\n",
                       i, j, cuCreal(val), cuCimag(val), diff);
                n_errors++;
            }
        }
    }

    if (n_errors == 0)
        printf("FFT computed successfully!\n");
}

// /* ------------------------------------------------------------------ */
// /* Helper: print NxN complex matrix from host                        */
// /* ------------------------------------------------------------------ */
// void print_matrix(const char *label,
//                   const std::vector<cuDoubleComplex> &M,
//                   int n)
// {
//     printf("\n%s:\n", label);
//     for (int r = 0; r < n; r++) {
//         for (int c = 0; c < n; c++)
//             printf("  (%6.2f,%6.2f)", cuCreal(M[r*n+c]), cuCimag(M[r*n+c]));
//         printf("\n");
//     }
// }



// /* ------------------------------------------------------------------ */
// /* Global-memory butterfly (one kernel launch per stage)              */
// /* ------------------------------------------------------------------ */
// __global__ void kernel_butterfly(cuDoubleComplex *A, const cuDoubleComplex *twiddles, unsigned int n, unsigned int m, unsigned int step)
// {
//     /* map thread id back to the original (k, j) loop indices */
//     unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int half = m / 2;

//     /* total butterflies per stage = (n/m) groups * (m/2) per group */
//     if (tid < n / 2){
//         unsigned int k = (tid / half) * m;   /* outer loop: k += m        */
//         unsigned int j =  tid % half;        /* inner loop: j < m/2       */

//         cuDoubleComplex w = twiddles[j * step];
//         cuDoubleComplex t = cuCmul(w, A[k + j + half]);
//         cuDoubleComplex u = A[k + j];

//         A[k + j]        = cuCadd(u, t);
//         A[k + j + half] = cuCsub(u, t);
//     }
// }


// /* ------------------------------------------------------------------ */
// /* Shared-memory butterfly (all stages in SMEM — avoids DRAM traffic) */
// /* ------------------------------------------------------------------ */
// __global__ void kernel_butterfly_smem(
//     cuDoubleComplex * __restrict__ A,
//     const cuDoubleComplex * __restrict__ twiddles,
//     unsigned int n, unsigned int start_stage, unsigned int log_n)
// {
//     extern __shared__ cuDoubleComplex smem[];
//     unsigned int tid = threadIdx.x;

//     for (unsigned int i = tid; i < n; i += blockDim.x)
//         smem[i] = A[i];
//     __syncthreads();

//     for (unsigned int s = start_stage; s <= log_n; s++) {
//         unsigned int half = 1u << (s - 1);
//         unsigned int step = n >> s;
//         for (unsigned int i = tid; i < n / 2; i += blockDim.x) {
//             unsigned int group = i / half;
//             unsigned int j     = i % half;
//             unsigned int k     = group * (half * 2) + j;
//             cuDoubleComplex w = twiddles[j * step];
//             cuDoubleComplex t = cuCmul(w, smem[k + half]);
//             cuDoubleComplex u = smem[k];
//             smem[k]        = cuCadd(u, t);
//             smem[k + half] = cuCsub(u, t);
//         }
//         __syncthreads();
//     }

//     for (unsigned int i = tid; i < n; i += blockDim.x)
//         A[i] = smem[i];
// }

// /* ------------------------------------------------------------------ */
// /* Persistent device buffers                                          */
// /* ------------------------------------------------------------------ */
// static cuDoubleComplex *d_A = NULL, *d_tmp = NULL, *d_twiddle = NULL;
// static size_t buf_n = 0, tmp_n = 0, tw_n = 0;

// static void ensure_buffers(size_t n) {
//     if (buf_n < n) {
//         cudaFree(d_A);
//         CUDA_CHECK(cudaMalloc(&d_A, n * sizeof(cuDoubleComplex)));
//         buf_n = n;
//     }
//     if (tmp_n < n) {
//         cudaFree(d_tmp);
//         CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(cuDoubleComplex)));
//         tmp_n = n;
//     }
//     if (tw_n < n / 2) {
//         cudaFree(d_twiddle);
//         CUDA_CHECK(cudaMalloc(&d_twiddle, (n/2) * sizeof(cuDoubleComplex)));
//         unsigned int half = (unsigned int)(n / 2);
//         unsigned int nblk = (half + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//         kernel_precompute_twiddles<<<nblk, THREADS_PER_BLOCK>>>(
//             d_twiddle, half, 1.0 / (double)n);
//         CUDA_CHECK(cudaGetLastError());
//         tw_n = n / 2;
//     }
// }

// /* ------------------------------------------------------------------ */
// /* Public API — drop-in for iterative_fft()                           */
// /* ------------------------------------------------------------------ */
// void iterative_fft_cuda(const double _Complex *a, double _Complex *A, size_t n)
// {
//     if (n == 0 || (n & (n-1)) != 0) {
//         fprintf(stderr, "iterative_fft_cuda: n must be power of two\n");
//         return;
//     }
//     unsigned int N = (unsigned int)n, log_n = 0;
//     { unsigned int t = N; while (t >>= 1) log_n++; }

//     ensure_buffers(n);

//     CUDA_CHECK(cudaMemcpy(d_A, a, n * sizeof(cuDoubleComplex),
//                           cudaMemcpyHostToDevice));

//     /* Bit-reverse into d_tmp, then swap so d_A holds bit-reversed data */
//     unsigned int nblk = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//     kernel_bit_reverse_copy<<<nblk, THREADS_PER_BLOCK>>>(d_A, d_tmp, N, log_n);
//     CUDA_CHECK(cudaGetLastError());
//     { cuDoubleComplex *s = d_A; d_A = d_tmp; d_tmp = s; }

//     /* FFT stages */
//     if (n <= (size_t)(2 * MAX_SHARED_ELEMS)) {
//         /* Small n: everything in shared memory, single launch */
//         kernel_butterfly_smem<<<1, THREADS_PER_BLOCK,
//                                  n * sizeof(cuDoubleComplex)>>>(
//             d_A, d_twiddle, N, 1, log_n);
//         CUDA_CHECK(cudaGetLastError());
//     } else {
//         /* Large n: global-memory butterflies for all stages */
//         for (unsigned int s = 1; s <= log_n; s++) {
//             unsigned int half = 1u << (s - 1);
//             unsigned int step = N >> s;
//             nblk = (N/2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//             kernel_butterfly<<<nblk, THREADS_PER_BLOCK>>>(
//                 d_A, d_twiddle, N, half, step);
//         CUDA_CHECK(cudaGetLastError());
//         }
//     }

//     CUDA_CHECK(cudaMemcpy(A, d_A, n * sizeof(cuDoubleComplex),
//                           cudaMemcpyDeviceToHost));
// }

// void iterative_fft_cuda_cleanup(void) {
//     cudaFree(d_A);       d_A       = NULL; buf_n = 0;
//     cudaFree(d_tmp);     d_tmp     = NULL; tmp_n = 0;
//     cudaFree(d_twiddle); d_twiddle = NULL; tw_n  = 0;
// }                        