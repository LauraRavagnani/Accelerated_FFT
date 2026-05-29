/*
 * fft_cuda.cu  —  Iterative Cooley–Tukey FFT, CUDA edition
 *
 * Target: Jetson Nano 4 GB — Maxwell GM20B, sm_53, 128 CUDA cores
 *
 * Build:
 *   nvcc -O3 -arch=sm_53 -use_fast_math -o fft_cuda fft_cuda.cu -lm
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// #define CUDA_CHECK(call) do { \
//     cudaError_t _e = (call); \
//     if (_e != cudaSuccess) { \
//         fprintf(stderr, "CUDA error %s:%d  %s\n", \
//                 __FILE__, __LINE__, cudaGetErrorString(_e)); \
//         exit(EXIT_FAILURE); \
//     } \
// } while (0)

#define THREADS_PER_BLOCK  2
#define N 8

/* ------------------------------------------------------------------ */
/* Bit-reversal kernel */
/* ------------------------------------------------------------------ */
__global__ void kernel_bit_reverse_copy(const cuDoubleComplex *a, cuDoubleComplex *A, unsigned int n, unsigned int n_bits)
{
    auto k = blockIdx.x * blockDim.x + threadIdx.x;  
    if (k < n){     //one element per thread
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
/* Twiddle kernel  */ 
/* ------------------------------------------------------------------ */
__global__ void kernel_precompute_twiddles(cuDoubleComplex *tw, unsigned int n)
{
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n/2){   //need only half threads(one element per thread)  
        double s, c;
        sincos(-2.0 * M_PI * (double)k / (double)n, &s, &c);
        tw[k] = make_cuDoubleComplex(c, s);
    }   
}




/* ------------------------------------------------------------------ */
/* Global-memory butterfly (one kernel launch per stage)              */
/* ------------------------------------------------------------------ */
__global__ void kernel_butterfly(cuDoubleComplex *A, const cuDoubleComplex *twiddles, unsigned int n, unsigned int m, unsigned int step)
{
    /* map thread id back to the original (k, j) loop indices */
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half = m / 2;

    /* total butterflies per stage = (n/m) groups * (m/2) per group */
    if (tid < n / 2){
        unsigned int k = (tid / half) * m;   /* outer loop: k += m        */
        unsigned int j =  tid % half;        /* inner loop: j < m/2       */

        cuDoubleComplex w = twiddles[j * step];
        cuDoubleComplex t = cuCmul(w, A[k + j + half]);
        cuDoubleComplex u = A[k + j];

        A[k + j]        = cuCadd(u, t);
        A[k + j + half] = cuCsub(u, t);
    }
}

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