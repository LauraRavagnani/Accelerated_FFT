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
/* Global-memory butterfly (one kernel launch per stage)             */
/* ------------------------------------------------------------------ */
__global__ void kernel_butterfly(cuDoubleComplex *A,
                                  const cuDoubleComplex *twiddles,
                                  unsigned int n,
                                  unsigned int m,
                                  unsigned int step)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half = m / 2;
    if (tid < n / 2) {
        unsigned int k = (tid / half) * m;
        unsigned int j =  tid % half;
        cuDoubleComplex w = twiddles[j * step];
        cuDoubleComplex t = cuCmul(w, A[k + j + half]);
        cuDoubleComplex u = A[k + j];
        A[k + j]        = cuCadd(u, t);
        A[k + j + half] = cuCsub(u, t);
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    size_t n_bits = (size_t)log2((double)N);
    size_t log_n  = n_bits;

    std::vector<cuDoubleComplex> a(N), A(N), tw(N/2);
    for (int i = 0; i < N; i++){
        a[i] = make_cuDoubleComplex(cos(2.0 * M_PI * 2.0 * (double)i / (double)N), 0.0);
    }
    
    /* ---- device allocations ---- */
    cuDoubleComplex *d_a, *d_A, *d_tw;
    size_t size    = N     * sizeof(cuDoubleComplex);
    size_t size_tw = (N/2) * sizeof(cuDoubleComplex);

    cudaMalloc((void**)&d_a,  size);
    cudaMalloc((void**)&d_A,  size);
    cudaMalloc((void**)&d_tw, size_tw);

    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);

    /* ---- print input ---- */
    printf("Input vector a:\n");
    for (int i = 0; i < N; i++)
        printf("  a[%d] = (%6.2f, %6.2f)\n", i, cuCreal(a[i]), cuCimag(a[i]));

    /* ---- step 1: twiddle factors ---- */
    int N_b   = (N/2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_precompute_twiddles<<<N_b, THREADS_PER_BLOCK>>>(d_tw, N);
    cudaDeviceSynchronize();

    /* print twiddles */
    cudaMemcpy(tw.data(), d_tw, size_tw, cudaMemcpyDeviceToHost);
    printf("\nTwiddle factors (n=%d):\n", N);
    for (int i = 0; i < N/2; i++)
        printf("  tw[%d] = (%7.4f, %7.4f)\n", i, cuCreal(tw[i]), cuCimag(tw[i]));

    /* ---- step 2: bit-reverse copy ---- */
    N_b = N / THREADS_PER_BLOCK;
    kernel_bit_reverse_copy<<<N_b, THREADS_PER_BLOCK>>>(d_a, d_A, N, n_bits);
    cudaDeviceSynchronize();

    /* print bit-reversed array */
    cudaMemcpy(A.data(), d_A, size, cudaMemcpyDeviceToHost);
    printf("\nAfter bit-reverse copy:\n");
    for (int i = 0; i < N; i++)
        printf("  A[%d] = (%6.2f, %6.2f)\n", i, cuCreal(A[i]), cuCimag(A[i]));

    /* ---- step 3: FFT stages ---- */
    for (size_t s = 1; s <= log_n; s++) {
        unsigned int m    = 1u << s;          /* 2^s                  */
        unsigned int step = N / m;

        N_b = (N/2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel_butterfly<<<N_b, THREADS_PER_BLOCK>>>(d_A, d_tw, N, m, step);
        cudaDeviceSynchronize();

        /* print after each stage */
        cudaMemcpy(A.data(), d_A, size, cudaMemcpyDeviceToHost);
        printf("\nAfter stage s=%zu (m=%u, step=%u):\n", s, m, step);
        for (int i = 0; i < N; i++)
            printf("  A[%d] = (%8.4f, %8.4f)\n", i, cuCreal(A[i]), cuCimag(A[i]));
    }

    /* ---- final result ---- */
    printf("\nFFT result:\n");
    for (int i = 0; i < N; i++)
        printf("  X[%d] = (%8.4f, %8.4f)\n", i, cuCreal(A[i]), cuCimag(A[i]));

    /* ---- cleanup ---- */
    cudaFree(d_a);
    cudaFree(d_A);
    cudaFree(d_tw);

    return 0;
}