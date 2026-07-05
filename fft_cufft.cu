/////////////////////////////////////////////////////////////////
/////    nvcc -O2 fft_cufft.cu -lcufft -o fft_cufft.out     /////
/////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <cufft.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "utils_cuda_global.cu"

int main(int argc, char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <Size N>\n", argv[0]);
        return 1;
    }
    
    int n  = atoi(argv[1]);

    const int    N  = n;     
    const size_t fx = 2;       
    const size_t fy = 3;       
    
    cudaEvent_t ev_plan_done, ev_fft_done;
    
    cudaEventCreate(&ev_plan_done);
    cudaEventCreate(&ev_fft_done);

    // ---------------------------------------------------------------------- //
    // Host allocation and dataset creation                                   //
    // ---------------------------------------------------------------------- //
    struct dataset data = create_dataset(N, fx, fy);

    const size_t size = N * N * sizeof(cuDoubleComplex);


    // ---------------------------------------------------------------------- //
    // Device allocation                                                      //
    // ---------------------------------------------------------------------- //
    cuDoubleComplex *d_Gxy = nullptr;
    cudaMalloc((void**)&d_Gxy, size);

    cudaMemcpy(d_Gxy, data.Gxy, size, cudaMemcpyHostToDevice);

    // Create and execute cuFFT plan 
    cufftHandle plan;
    cufftPlan2d(&plan, N, N, CUFFT_Z2Z);
    cudaEventRecord(ev_plan_done);

    // Forward transform, in-place
    cufftExecZ2Z(plan, d_Gxy, d_Gxy, CUFFT_FORWARD); // double precision
    cudaEventRecord(ev_fft_done);

    // ---------------------------------------------------------------------- //
    // Copy result back                                                       //
    // ---------------------------------------------------------------------- //
    cuDoubleComplex *h_result = (cuDoubleComplex*)malloc(size);
    cudaMemcpy(h_result, d_Gxy, size, cudaMemcpyDeviceToHost);

    float time_fft = 0.0f;

    cudaEventElapsedTime(&time_fft, ev_plan_done, ev_fft_done);
    
    printf("%.9f\n", time_fft * 1e-3);
    
    cufftDestroy(plan);
    cudaFree(d_Gxy);
    cudaEventDestroy(ev_plan_done);
    cudaEventDestroy(ev_fft_done);
    free(data.grid_x);
    free(data.grid_y);
    free(data.Gxy);
    free(h_result);

    return 0;
}