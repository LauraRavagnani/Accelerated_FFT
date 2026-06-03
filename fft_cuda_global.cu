#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include "utils_serial.c"
#include "utils_cuda_naif.cu"

//#define THREADS_PER_BLOCK  2
#define N 512

/* ------------------------------------------------------------------ */
/* Helper: FFT on every row of a NxN matrix on device                */
/* ------------------------------------------------------------------ */
// void cuda_fft(cuDoubleComplex *d_mat,
//               cuDoubleComplex *d_tmp,
//               cuDoubleComplex *d_tw,
//               int n, int n_bits, int THREADS_PER_BLOCK)
// {
//     int N_b_row = n / THREADS_PER_BLOCK;

//     for (int row = 0; row < n; row++) {
//         unsigned int row_offset = row * n;

//         /* bit-reverse copy: read from matrix row, write to temp buffer */
//         kernel_bit_reverse_copy<<<N_b_row, THREADS_PER_BLOCK>>>(d_mat + row_offset, d_tmp, n, n_bits);
//         cudaDeviceSynchronize();

//         /* copy bit-reversed result back into the matrix row */
//         cudaMemcpy(d_mat + row_offset, d_tmp, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

//         /* butterfly stages */
//         for (int s = 1; s <= n_bits; s++) {
//             unsigned int m    = 1u << s;
//             unsigned int step = n / m;
//             int N_b = (n/2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//             kernel_butterfly<<<N_b, THREADS_PER_BLOCK>>>(d_mat, d_tw, n, m, step, row_offset);
//             cudaDeviceSynchronize();
//         }
//     }
// }

void cuda_fft(cuDoubleComplex *d_mat,
              cuDoubleComplex *d_tmp,
              cuDoubleComplex *d_tw,
              int n, int n_bits, dim3 block, dim3 grid_br, dim3 grid_bf)
{
    // // 2D grid: x = blocks along row, y = one block per row
    // dim3 block(TPB);
    // dim3 grid_br((n     + TPB - 1) / TPB, n);  // bit-reverse: n elements per row
    // dim3 grid_bf((n/2   + TPB - 1) / TPB, n);  // butterfly:   n/2 pairs per row

    // Step 1: bit-reverse all rows at once
    kernel_bit_reverse_copy<<<grid_br, block>>>(d_mat, d_tmp, n, n_bits);
    cudaDeviceSynchronize();

    // copy bit-reversed result back (all rows at once)
    cudaMemcpy(d_mat, d_tmp, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

    // Step 2: butterfly stages — one launch per stage, all rows in parallel
    for (int s = 1; s <= n_bits; s++) {
        unsigned int m    = 1u << s;
        unsigned int step = n / m;
        kernel_butterfly<<<grid_bf, block>>>(d_mat, d_tw, n, m, step);
    }
    cudaDeviceSynchronize();
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[]){
    int THREADS_PER_BLOCK  = atoi(argv[1]);

    int    n_bits = (int)log2((double)N);
    size_t size   = N * N * sizeof(cuDoubleComplex);
    //size_t size_r = N     * sizeof(cuDoubleComplex);
    size_t size_tw= (N/2) * sizeof(cuDoubleComplex);

    size_t fx = 2, fy = 2;

    int tpb2d = min(THREADS_PER_BLOCK, 32); // keep 2D block ≤ 32×32 = 1024

    dim3 block2d(tpb2d, tpb2d);
    dim3 grid2d(N / tpb2d, N / tpb2d);

    // 2D grid: x = blocks along row, y = one block per row
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid_br((N     + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, N);  // bit-reverse: n elements per row
    dim3 grid_bf((N/2   + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, N);  // butterfly:   n/2 pairs per row

    
    /* ---- host allocation ----*/
    cuDoubleComplex *h_X_kl   = (cuDoubleComplex*)malloc(size);

    struct dataset data = create_dataset_cuda(N, fx, fy);

    /* ---- device allocations ---- */
    cuDoubleComplex *d_Gxy, *d_Gxy_T, *d_X_k, *d_tw;
    cudaMalloc((void**)&d_Gxy,   size);
    cudaMalloc((void**)&d_Gxy_T, size);
    cudaMalloc((void**)&d_X_k,   size);
    cudaMalloc((void**)&d_tw,    size_tw);

    cudaMemcpy(d_Gxy, data.Gxy, size, cudaMemcpyHostToDevice);

    /* ---- precompute twiddles ---- */
    int N_b_tw = (N/2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    kernel_precompute_twiddles<<<N_b_tw, THREADS_PER_BLOCK>>>(d_tw, N);
    cudaDeviceSynchronize();

    /* ---- step 1: FFT on each row ---- */
    cuda_fft(d_Gxy, d_X_k, d_tw, N, n_bits, block, grid_br, grid_bf);

    //cudaMemcpy(data.Gxy, d_Gxy, size, cudaMemcpyDeviceToHost);

    /* ---- step 2: transpose ---- */
    matrixTransposition<<<grid2d, block2d>>>(d_Gxy, d_Gxy_T, N);
    cudaDeviceSynchronize();

    /* ---- step 3: FFT on each row of transposed matrix ---- */
    cuda_fft(d_Gxy_T, d_X_k, d_tw, N, n_bits, block, grid_br, grid_bf);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("%.9f\n", elapsed*1e-6);
    
    /* ---- step 4: transpose back to get final 2D FFT ---- */
    // matrixTransposition<<<grid2d, block2d>>>(d_Gxy_T, d_Gxy, N);
    // cudaDeviceSynchronize();

    /* ---- copy final result to host and print ---- */
    cudaMemcpy(h_X_kl, d_Gxy_T, size, cudaMemcpyDeviceToHost);
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