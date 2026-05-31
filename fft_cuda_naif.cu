#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include "utils_serial.c"
#include "utils_cuda_naif.cu"

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    int    n_bits = (int)log2((double)N);
    size_t size   = N * N * sizeof(cuDoubleComplex);
    size_t size_r = N     * sizeof(cuDoubleComplex);
    size_t size_tw= (N/2) * sizeof(cuDoubleComplex);

    size_t fx = 2, fy = 2;
    
    /* ---- host allocation ----*/
    cuDoubleComplex *h_X_kl   = (cuDoubleComplex*)malloc(size);

    std::vector<cuDoubleComplex> h_mat(N * N);

    struct dataset data = create_dataset_cuda(N, fx, fy);

    /* ---- device allocations ---- */
    cuDoubleComplex *d_Gxy, *d_Gxy_T, *d_X_k, *d_tw;
    cudaMalloc((void**)&d_Gxy,   size);
    cudaMalloc((void**)&d_Gxy_T, size);
    cudaMalloc((void**)&d_X_k,   size_r);
    cudaMalloc((void**)&d_tw,    size_tw);

    cudaMemcpy(d_Gxy, data.Gxy, size, cudaMemcpyHostToDevice);

    /* ---- precompute twiddles ---- */
    int N_b_tw = (N/2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_precompute_twiddles<<<N_b_tw, THREADS_PER_BLOCK>>>(d_tw, N);
    cudaDeviceSynchronize();

    /* ---- step 1: FFT on each row ---- */
    cuda_fft(d_Gxy, d_X_k, d_tw, N, n_bits);

    cudaMemcpy(data.Gxy, d_Gxy, size, cudaMemcpyDeviceToHost);

    /* ---- step 2: transpose ---- */
    dim3 block2d(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid2d(N / THREADS_PER_BLOCK, N / THREADS_PER_BLOCK);
    matrixTransposition<<<grid2d, block2d>>>(d_Gxy, d_Gxy_T, N);
    cudaDeviceSynchronize();

    /* ---- step 3: FFT on each row of transposed matrix ---- */
    cuda_fft(d_Gxy_T, d_X_k, d_tw, N, n_bits);

    /* ---- step 4: transpose back to get final 2D FFT ---- */
    matrixTransposition<<<grid2d, block2d>>>(d_Gxy_T, d_Gxy, N);
    cudaDeviceSynchronize();

    /* ---- copy final result to host and print ---- */
    cudaMemcpy(h_X_kl, d_Gxy, size, cudaMemcpyDeviceToHost);
    // print_matrix("Final 2D FFT result", h_mat, N);

    validate_fft(h_X_kl, N, fx, fy);

    /* ---- cleanup ---- */
    cudaFree(d_Gxy);
    cudaFree(d_Gxy_T);
    cudaFree(d_X_k);
    cudaFree(d_tw);

    free(h_X_kl);

    return 0;
}