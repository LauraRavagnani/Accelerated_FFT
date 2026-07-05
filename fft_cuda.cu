///////////////////////////////////////////////////////////////
/////   nvcc -O2 -arch=sm_75 -o fft_cuda.out fft_cuda.cu  /////
///////////////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils_cuda.cu"

// -----------------------------------------------------------------------------
// Run one full 1-D FFT pass over every row of d_Gxy
//   - one kernel_bit_reverse launch
//   - log2(n) kernel_butterfly launches
//   - one cudaDeviceSynchronize() at the end of the butterfly loop
// -----------------------------------------------------------------------------
static void fft_rows(cuDoubleComplex *d_mat, unsigned int n, unsigned int n_bits){
    int THREADS_PER_BLOCK = 256;
    
    // each thread handles one element
    //   grid.x = position within the row 
    //   grid.y = row index 
    dim3 n_blocks_br(n / THREADS_PER_BLOCK, n);
    kernel_bit_reverse<<<n_blocks_br, THREADS_PER_BLOCK>>>(d_mat, n, n_bits);
    cudaDeviceSynchronize();

    // butterfly stages: each thread handles one pair
    dim3 n_blocks_bf((n / 2) / THREADS_PER_BLOCK, n);
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m = 1u << s;
        unsigned int step = n / m;  // twiddle stride: picks which n-th roots of unity this stage needs
        kernel_butterfly<<<n_blocks_bf, THREADS_PER_BLOCK>>>(d_mat, n, m, step);
        cudaDeviceSynchronize();    // stage s+1 reads pairs that stage s just wrote
    }
}

// Same as fft_rows(), but with THREADS_PER_BLOCK exposed as a parameter for tuning
static void fft_rows_tune(cuDoubleComplex *d_mat, unsigned int n, unsigned int n_bits, int THREADS_PER_BLOCK){
    dim3 n_blocks_br(n / THREADS_PER_BLOCK, n);
    kernel_bit_reverse<<<n_blocks_br, THREADS_PER_BLOCK>>>(d_mat, n, n_bits);
    cudaDeviceSynchronize();

    // butterfly stages: each thread handles one pair, 2D grid over all rows
    // dim3 blk_bf(THREADS_PER_BLOCK);
    dim3 n_blocks_bf((n / 2) / THREADS_PER_BLOCK, n);
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m    = 1u << s;
        unsigned int step = n / m;
        kernel_butterfly<<<n_blocks_bf, THREADS_PER_BLOCK>>>(d_mat, n, m, step);
        cudaDeviceSynchronize();  
    }
}

// ------------------------------------------- //
// Main                                        //
// ------------------------------------------- //
int main(int argc, char *argv[])
{
    bool tune = false;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <threads_per_block>\n", argv[0]);
        return 1;
    }
    
    int N  = atoi(argv[1]);
    
    const unsigned int n  = N;
    const unsigned int n_bits = (unsigned int)log2((double)n);
    const size_t       size   = n * n * sizeof(cuDoubleComplex);
    const size_t       fx = 2, fy = 3;

    // tile size for the transpose kernel (keep ≤ 32 so block ≤ 1024 threads)
    // a warp of 32 threads reads (or writes) 32 contiguous elements
    // of a row. A TILE larger than 32 would need >1024 threads per block (TILE*TILE),
    // which exceeds the hardware limit; 
    const int TILE = 32;

    // -------------------- //
    // Timing               //
    // -------------------- //
    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t HtD_start, HtD_stop;
    cudaEvent_t DtH_start, DtH_stop;
    
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventCreate(&HtD_start);
    cudaEventCreate(&HtD_stop);
    cudaEventCreate(&DtH_start);
    cudaEventCreate(&DtH_stop);

    // -------------------------------------------------------------------------
    // Host allocation and dataset creation
    // -------------------------------------------------------------------------
    cuDoubleComplex *h_result;
    cudaMallocHost((void**)&h_result, size);    // Allocating as pinned memory lets cudaMemcpy use direct DMA,
                                                // giving higher H2D/D2H bandwidth

    struct dataset data = create_dataset(n, fx, fy);
    cuDoubleComplex *h_Gxy;
    cudaMallocHost((void**)&h_Gxy, size);
    memcpy(h_Gxy, data.Gxy, size);

    // --------------------- //
    // Device allocation     //
    // --------------------- //
    cuDoubleComplex *d_Gxy, *d_Gxy_T;
    cudaMalloc((void**)&d_Gxy,   size);
    cudaMalloc((void**)&d_Gxy_T, size);

    cudaEventRecord(HtD_start);
    cudaMemcpy(d_Gxy, h_Gxy, size, cudaMemcpyHostToDevice);
    cudaEventRecord(HtD_stop);
    cudaEventSynchronize(HtD_stop);

    float time_HtD;
    cudaEventElapsedTime(&time_HtD, HtD_start, HtD_stop);

    // ------------------------------------------ //
    // Grid / block configs for the transpose     //
    // ------------------------------------------ //
    dim3 blk_tr(TILE, TILE);
    dim3 grd_tr(n / TILE, n / TILE);

    cudaEventRecord(ev_start);

    if(tune){
        int THREADS_PER_BLOCK  = atoi(argv[2]);
        fft_rows_tune(d_Gxy, n, n_bits, THREADS_PER_BLOCK);
    }
    else
        fft_rows(d_Gxy, n, n_bits);

    //   Transpose, so the same row-parallel, coalesced-access kernel can be reused
    kernel_transpose<<<grd_tr, blk_tr>>>(d_Gxy, d_Gxy_T, n);
    cudaDeviceSynchronize();

    if(tune){
        int THREADS_PER_BLOCK  = atoi(argv[2]);
        fft_rows_tune(d_Gxy_T, n, n_bits, THREADS_PER_BLOCK);
    }
    else
        fft_rows(d_Gxy_T, n, n_bits);

    // The above step stores the transpose results, so transpose back
    kernel_transpose<<<grd_tr, blk_tr>>>(d_Gxy_T, d_Gxy, n);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, ev_start, ev_stop);

    // --------------------------------- //
    // Copy result back and validate     //
    // --------------------------------- //
    cudaEventRecord(DtH_start);
    cudaMemcpy(h_result, d_Gxy, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(DtH_stop);

    cudaEventSynchronize(DtH_stop);

    float time_DtH;
    cudaEventElapsedTime(&time_DtH, DtH_start, DtH_stop);

    float total_time;
    cudaEventElapsedTime(&total_time, HtD_start, DtH_stop);
    
    validate_fft(h_result, n, fx, fy);

    // ------------------------------------ //
    // Measure Throughput and Bandwidth     //
    // ------------------------------------ //
    double compute_gflops = gflops(n, elapsed);      
    double htd_gbps        = gbps(size, time_HtD);              
    double dth_gbps        = gbps(size, time_DtH);             

 
    printf("%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\n",
            total_time * 1e-3,
            elapsed * 1e-3,
            time_HtD * 1e-3,
            time_DtH * 1e-3,
            compute_gflops,
            htd_gbps,
            dth_gbps
            );

    cudaFree(d_Gxy);
    cudaFree(d_Gxy_T);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaEventDestroy(DtH_start);
    cudaEventDestroy(DtH_stop);
    cudaEventDestroy(HtD_start);
    cudaEventDestroy(HtD_stop);
    cudaFreeHost(h_result);
    free(data.grid_x);
    free(data.grid_y);
    free(data.Gxy);
    cudaFreeHost(h_Gxy);

    return 0;
}
