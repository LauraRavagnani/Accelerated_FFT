// #include <vector>
// #include <cuda_runtime.h>
// #include <cuComplex.h>
// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>

// #define THREADS_PER_BLOCK  2
// #define N 8

// /* ------------------------------------------------------------------ */
// /* Bit-reversal kernel                                                */
// /* ------------------------------------------------------------------ */
// __global__ void kernel_bit_reverse_copy(const cuDoubleComplex *a,
//                                               cuDoubleComplex *A,
//                                         unsigned int n,
//                                         unsigned int n_bits)
// {
//     auto k = blockIdx.x * blockDim.x + threadIdx.x;
//     if (k < n) {
//         auto r   = 0;
//         auto tmp = k;
//         for (auto b = 0; b < n_bits; b++) {
//             r   = (r << 1) | (tmp & 1);
//             tmp >>= 1;
//         }
//         A[r] = a[k];
//     }
// }

// /* ------------------------------------------------------------------ */
// /* Twiddle kernel                                                     */
// /* ------------------------------------------------------------------ */
// __global__ void kernel_precompute_twiddles(cuDoubleComplex *tw, unsigned int n)
// {
//     unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
//     if (k < n / 2) {
//         double s, c;
//         sincos(-2.0 * M_PI * (double)k / (double)n, &s, &c);
//         tw[k] = make_cuDoubleComplex(c, s);
//     }
// }

// /* ------------------------------------------------------------------ */
// /* Global-memory butterfly (one kernel launch per stage)             */
// /* ------------------------------------------------------------------ */
// __global__ void kernel_butterfly(cuDoubleComplex *A,
//                                   const cuDoubleComplex *twiddles,
//                                   unsigned int n,
//                                   unsigned int m,
//                                   unsigned int step)
// {
//     unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int half = m / 2;
//     if (tid < n / 2) {
//         unsigned int k = (tid / half) * m;
//         unsigned int j =  tid % half;
//         cuDoubleComplex w = twiddles[j * step];
//         cuDoubleComplex t = cuCmul(w, A[k + j + half]);
//         cuDoubleComplex u = A[k + j];
//         A[k + j]        = cuCadd(u, t);
//         A[k + j + half] = cuCsub(u, t);
//     }
// }

// /* ------------------------------------------------------------------ */
// /* Main                                                               */
// /* ------------------------------------------------------------------ */
// int main(int argc, char** argv)
// {
//     size_t n_bits = (size_t)log2((double)N);
//     size_t log_n  = n_bits;

//     std::vector<cuDoubleComplex> a(N), A(N), tw(N/2);
//     for (int i = 0; i < N; i++){
//         a[i] = make_cuDoubleComplex(cos(2.0 * M_PI * 2.0 * (double)i / (double)N), 0.0);
//     }
    
//     /* ---- device allocations ---- */
//     cuDoubleComplex *d_a, *d_A, *d_tw;
//     size_t size    = N     * sizeof(cuDoubleComplex);
//     size_t size_tw = (N/2) * sizeof(cuDoubleComplex);

//     cudaMalloc((void**)&d_a,  size);
//     cudaMalloc((void**)&d_A,  size);
//     cudaMalloc((void**)&d_tw, size_tw);

//     cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);

//     /* ---- print input ---- */
//     printf("Input vector a:\n");
//     for (int i = 0; i < N; i++)
//         printf("  a[%d] = (%6.2f, %6.2f)\n", i, cuCreal(a[i]), cuCimag(a[i]));

//     /* ---- step 1: twiddle factors ---- */
//     int N_b   = (N/2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//     kernel_precompute_twiddles<<<N_b, THREADS_PER_BLOCK>>>(d_tw, N);
//     cudaDeviceSynchronize();

//     /* print twiddles */
//     cudaMemcpy(tw.data(), d_tw, size_tw, cudaMemcpyDeviceToHost);
//     printf("\nTwiddle factors (n=%d):\n", N);
//     for (int i = 0; i < N/2; i++)
//         printf("  tw[%d] = (%7.4f, %7.4f)\n", i, cuCreal(tw[i]), cuCimag(tw[i]));

//     /* ---- step 2: bit-reverse copy ---- */
//     N_b = N / THREADS_PER_BLOCK;
//     kernel_bit_reverse_copy<<<N_b, THREADS_PER_BLOCK>>>(d_a, d_A, N, n_bits);
//     cudaDeviceSynchronize();

//     /* print bit-reversed array */
//     cudaMemcpy(A.data(), d_A, size, cudaMemcpyDeviceToHost);
//     printf("\nAfter bit-reverse copy:\n");
//     for (int i = 0; i < N; i++)
//         printf("  A[%d] = (%6.2f, %6.2f)\n", i, cuCreal(A[i]), cuCimag(A[i]));

//     /* ---- step 3: FFT stages ---- */
//     for (size_t s = 1; s <= log_n; s++) {
//         unsigned int m    = 1u << s;          /* 2^s                  */
//         unsigned int step = N / m;

//         N_b = (N/2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//         kernel_butterfly<<<N_b, THREADS_PER_BLOCK>>>(d_A, d_tw, N, m, step);
//         cudaDeviceSynchronize();

//         /* print after each stage */
//         cudaMemcpy(A.data(), d_A, size, cudaMemcpyDeviceToHost);
//         printf("\nAfter stage s=%zu (m=%u, step=%u):\n", s, m, step);
//         for (int i = 0; i < N; i++)
//             printf("  A[%d] = (%8.4f, %8.4f)\n", i, cuCreal(A[i]), cuCimag(A[i]));
//     }

//     /* ---- final result ---- */
//     printf("\nFFT result:\n");
//     for (int i = 0; i < N; i++)
//         printf("  X[%d] = (%8.4f, %8.4f)\n", i, cuCreal(A[i]), cuCimag(A[i]));

//     /* ---- cleanup ---- */
//     cudaFree(d_a);
//     cudaFree(d_A);
//     cudaFree(d_tw);

//     return 0;
// }

#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK  2
#define N 8

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

/* ------------------------------------------------------------------ */
/* Helper: print NxN complex matrix from host                        */
/* ------------------------------------------------------------------ */
void print_matrix(const char *label,
                  const std::vector<cuDoubleComplex> &M,
                  int n)
{
    printf("\n%s:\n", label);
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++)
            printf("  (%6.2f,%6.2f)", cuCreal(M[r*n+c]), cuCimag(M[r*n+c]));
        printf("\n");
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    int    n_bits = (int)log2((double)N);
    size_t size   = N * N * sizeof(cuDoubleComplex);
    size_t size_r = N     * sizeof(cuDoubleComplex);
    size_t size_tw= (N/2) * sizeof(cuDoubleComplex);

    /* ---- build 2D cosine input: cos(2π*fx*col/N + 2π*fy*row/N) ---- */
    int fx = 2, fy = 2;
    std::vector<cuDoubleComplex> h_mat(N * N);
    for (int r = 0; r < N; r++)
    for (int c = 0; c < N; c++)
        h_mat[r*N+c] = make_cuDoubleComplex(
            cos(2.0*M_PI*fx*c/N) * cos(2.0*M_PI*fy*r/N), 0.0);

    print_matrix("Input matrix", h_mat, N);

    /* ---- device allocations ---- */
    cuDoubleComplex *d_mat, *d_mat_T, *d_tmp, *d_tw;
    cudaMalloc((void**)&d_mat,   size);
    cudaMalloc((void**)&d_mat_T, size);
    cudaMalloc((void**)&d_tmp,   size_r);
    cudaMalloc((void**)&d_tw,    size_tw);

    cudaMemcpy(d_mat, h_mat.data(), size, cudaMemcpyHostToDevice);

    /* ---- precompute twiddles ---- */
    int N_b_tw = (N/2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_precompute_twiddles<<<N_b_tw, THREADS_PER_BLOCK>>>(d_tw, N);
    cudaDeviceSynchronize();

    /* ---- step 1: FFT on each row ---- */
    cuda_fft(d_mat, d_tmp, d_tw, N, n_bits);

    cudaMemcpy(h_mat.data(), d_mat, size, cudaMemcpyDeviceToHost);
    print_matrix("After row FFTs", h_mat, N);

    /* ---- step 2: transpose ---- */
    dim3 block2d(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid2d(N / THREADS_PER_BLOCK, N / THREADS_PER_BLOCK);
    matrixTransposition<<<grid2d, block2d>>>(d_mat, d_mat_T, N);
    cudaDeviceSynchronize();

    /* ---- step 3: FFT on each row of transposed matrix ---- */
    cuda_fft(d_mat_T, d_tmp, d_tw, N, n_bits);

    /* ---- step 4: transpose back to get final 2D FFT ---- */
    matrixTransposition<<<grid2d, block2d>>>(d_mat_T, d_mat, N);
    cudaDeviceSynchronize();

    /* ---- copy final result to host and print ---- */
    cudaMemcpy(h_mat.data(), d_mat, size, cudaMemcpyDeviceToHost);
    print_matrix("Final 2D FFT result", h_mat, N);

    printf("\nExpected peaks at (row,col) = (2,2),(2,6),(6,2),(6,6)\n");
    printf("with magnitude N*N/4 = %d each\n", N*N/4);

    /* ---- cleanup ---- */
    cudaFree(d_mat);
    cudaFree(d_mat_T);
    cudaFree(d_tmp);
    cudaFree(d_tw);

    return 0;
}