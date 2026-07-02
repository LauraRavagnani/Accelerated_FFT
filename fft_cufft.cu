// fft2d_cufft.cu
//
// Computes the 2D FFT of
//      G(x, y) = cos(2*pi*fx*x) * cos(2*pi*fy*y)
// with fx = fy = 2, using cuFFT (cufftExecZ2Z, double precision complex).
//
// The spatial domain is sampled on an N x N grid covering [0, 1) x [0, 1)
// and the FFT output is shifted so the zero-frequency component sits at the
// center of the array (like numpy's fftshift), making the four expected
// peaks at (+-fx, +-fy) easy to find.
//
// Build:
//   nvcc -O2 fft_cufft.cu -lcufft -o fft_cufft.out


#include <cuda_runtime.h>
#include <cufft.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "utils_cuda_global.cu"

// ---- Error-checking helpers -------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                             \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CUFFT_CHECK(call)                                                      \
    do {                                                                      \
        cufftResult err = (call);                                             \
        if (err != CUFFT_SUCCESS) {                                           \
            std::fprintf(stderr, "cuFFT error %s:%d: code %d\n", __FILE__,    \
                         __LINE__, (int)err);                                 \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                     \
    } while (0)

int main(int argc, char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <Size N>\n", argv[0]);
        return 1;
    }
    // const int THREADS_PER_BLOCK = 512;
    
    int n  = atoi(argv[1]);

    // ---------------- Parameters ----------------
    const int    N  = n;      // grid size (N x N), power of 2 is fastest for cuFFT
    const size_t fx = 2;        // spatial frequency in x (cycles per unit length)
    const size_t fy = 2;        // spatial frequency in y (cycles per unit length)
    
    cudaEvent_t ev_plan_done, ev_fft_done;
    
    CUDA_CHECK(cudaEventCreate(&ev_plan_done));
    CUDA_CHECK(cudaEventCreate(&ev_fft_done));


    // ---------------- Build input signal on host ----------------
    struct dataset data = create_dataset(N, fx, fy);


    // ---------------- Allocate device memory and copy input over ----------------
    const size_t size = N * N * sizeof(cuDoubleComplex);

    cuDoubleComplex *d_A = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));

    CUDA_CHECK(cudaMemcpy(d_A, data.Gxy, size, cudaMemcpyHostToDevice));

    // ---------------- Create and execute cuFFT plan ----------------
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, N, N, CUFFT_Z2Z));
    CUDA_CHECK(cudaEventRecord(ev_plan_done));

    // Forward transform, in-place
    CUFFT_CHECK(cufftExecZ2Z(plan, d_A, d_A, CUFFT_FORWARD)); // double precision
    CUDA_CHECK(cudaEventRecord(ev_fft_done));

    // ---------------- Copy result back to host ----------------
    cuDoubleComplex *h_result = (cuDoubleComplex*)malloc(size);
    CUDA_CHECK(cudaMemcpy(h_result, d_A, size, cudaMemcpyDeviceToHost));

    // ---------------- Read back timings ----------------
    float ms_fft = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&ms_fft, ev_plan_done, ev_fft_done));
    

    printf("%.9f\n", ms_fft * 1e-3);

    
    // ---------------- Cleanup GPU resources ----------------
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_A));
    
    CUDA_CHECK(cudaEventDestroy(ev_plan_done));
    CUDA_CHECK(cudaEventDestroy(ev_fft_done));

    // ---------------- Cleanup host resources ----------------
    free(data.grid_x);
    free(data.grid_y);
    free(data.Gxy);
    free(h_result);

    return 0;
}