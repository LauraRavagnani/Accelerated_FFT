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
    // const double Lx = 1.0;      // physical extent of domain in x: [0, Lx)
    // const double Ly = 1.0;      // physical extent of domain in y: [0, Ly)
    // const double dx = Lx / N;   // sample spacing in x
    // const double dy = Ly / N;   // sample spacing in y

    // std::printf("Grid: %d x %d, fx=%zu, fy=%zu, dx=%.5f, dy=%.5f\n", N, N, fx,
    //             fy, dx, dy);

    // ---------------- Timing setup ----------------
    // CUDA events give accurate GPU-side timing (unlike wrapping a CPU clock
    // around an async call, which would only measure launch overhead, not
    // actual device execution time).
    // cudaEvent_t ev_start, ev_h2d_done, ev_plan_done, ev_fft_done, ev_d2h_done;
    cudaEvent_t ev_plan_done, ev_fft_done;
    // CUDA_CHECK(cudaEventCreate(&ev_start));
    // CUDA_CHECK(cudaEventCreate(&ev_h2d_done));
    CUDA_CHECK(cudaEventCreate(&ev_plan_done));
    CUDA_CHECK(cudaEventCreate(&ev_fft_done));
    // CUDA_CHECK(cudaEventCreate(&ev_d2h_done));

    // auto host_t0 = std::chrono::high_resolution_clock::now();

    // ---------------- Build input signal on host ----------------
    struct dataset data = create_dataset(N, fx, fy);

    // auto host_t1 = std::chrono::high_resolution_clock::now();
    // double host_setup_ms =
    //     std::chrono::duration<double, std::milli>(host_t1 - host_t0).count();

    // ---------------- Allocate device memory and copy input over ----------------
    const size_t size = N * N * sizeof(cuDoubleComplex);

    cuDoubleComplex *d_A = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));

    // CUDA_CHECK(cudaEventRecord(ev_start));
    CUDA_CHECK(cudaMemcpy(d_A, data.Gxy, size, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaEventRecord(ev_h2d_done));

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
    // CUDA_CHECK(cudaEventRecord(ev_d2h_done));
    // CUDA_CHECK(cudaEventSynchronize(ev_d2h_done));

    // ---------------- Read back timings ----------------
    float ms_fft = 0.0f;
    // float ms_h2d = 0.0f, ms_plan = 0.0f, ms_fft = 0.0f, ms_d2h = 0.0f, ms_total_gpu = 0.0f;
    // CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ev_start, ev_h2d_done));
    // CUDA_CHECK(cudaEventElapsedTime(&ms_plan, ev_h2d_done, ev_plan_done));
    CUDA_CHECK(cudaEventElapsedTime(&ms_fft, ev_plan_done, ev_fft_done));
    // CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ev_fft_done, ev_d2h_done));
    // CUDA_CHECK(cudaEventElapsedTime(&ms_total_gpu, ev_start, ev_d2h_done));
    

    printf("%.9f\n", ms_fft * 1e-3);

    // std::printf("\n---------------- Timing ----------------\n");
    // std::printf("%-28s %10.4f ms\n", "Host dataset setup:", host_setup_ms);
    // std::printf("%-28s %10.4f ms\n", "H2D memcpy:", ms_h2d);
    // std::printf("%-28s %10.4f ms\n", "cuFFT plan creation:", ms_plan);
    // std::printf("%-28s %10.4f ms\n", "cuFFT execution (FFT):", ms_fft);
    // std::printf("%-28s %10.4f ms\n", "D2H memcpy:", ms_d2h);
    // std::printf("%-28s %10.4f ms\n", "Total GPU pipeline:", ms_total_gpu);
    // std::printf("------------------------------------------\n");

    // ---------------- Cleanup GPU resources ----------------
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_A));
    // CUDA_CHECK(cudaEventDestroy(ev_start));
    // CUDA_CHECK(cudaEventDestroy(ev_h2d_done));
    CUDA_CHECK(cudaEventDestroy(ev_plan_done));
    CUDA_CHECK(cudaEventDestroy(ev_fft_done));
    // CUDA_CHECK(cudaEventDestroy(ev_d2h_done));

    // ---------------- Post-process: magnitude + fftshift ----------------
    // cuFFT (like FFTW) outputs frequencies in the order:
    //   [0, 1, 2, ..., N/2-1, -N/2, ..., -1]  (for each dimension)
    // fftshift reorders this to:
    //   [-N/2, ..., -1, 0, 1, ..., N/2-1]
    // so that the zero-frequency (DC) term sits at the center of the array,
    // which matches typical visualizations (e.g., numpy.fft.fftshift).
    // std::vector<double> magnitude(N * N);
    // std::vector<double> shifted(N * N);

    // for (int row = 0; row < N; ++row) {
    //     for (int col = 0; col < N; ++col) {
    //         cuDoubleComplex c = h_result[row * N + col];
    //         // Normalize by N*N so magnitude matches the analytic amplitude
    //         // (un-normalized cuFFT forward transform scales output by N*N).
    //         double re = c.x / (double)(N * N);
    //         double im = c.y / (double)(N * N);
    //         magnitude[row * N + col] = std::sqrt(re * re + im * im);
    //     }
    // }

    // int half = N / 2;
    // for (int row = 0; row < N; ++row) {
    //     int srow = (row + half) % N;
    //     for (int col = 0; col < N; ++col) {
    //         int scol = (col + half) % N;
    //         shifted[srow * N + scol] = magnitude[row * N + col];
    //     }
    // }

    // // Frequency axis after shift: index i (0..N-1) maps to frequency
    // //   freq(i) = (i - N/2) / (N * dx)   [cycles per unit length]
    // auto freq_axis = [&](int i, double d) {
    //     return (i - half) / (N * d);
    // };

    // ---------------- Report the dominant peaks ----------------
    // For G(x,y) = cos(2*pi*fx*x)*cos(2*pi*fy*y), the analytic 2D FT is four
    // delta functions at (u, v) = (+-fx, +-fy), each with amplitude 1/4.
    // std::printf("\nTop magnitude peaks in the shifted spectrum (u = freq_x, v = freq_y):\n");
    // std::printf("%-10s %-10s %-12s\n", "u", "v", "magnitude");

    // // Simple approach: scan the whole shifted grid, print entries above a
    // // threshold (here, anything within 1% of the global max).
    // double max_mag = 0.0;
    // for (double m : shifted) max_mag = std::max(max_mag, m);
    // double threshold = 0.01 * max_mag;

    // for (int row = 0; row < N; ++row) {
    //     double v = freq_axis(row, dy);
    //     for (int col = 0; col < N; ++col) {
    //         double u = freq_axis(col, dx);
    //         double m = shifted[row * N + col];
    //         if (m > threshold) {
    //             std::printf("%-10.3f %-10.3f %-12.6f\n", u, v, m);
    //         }
    //     }
    // }

    // ---------------- Write full spectrum to CSV for plotting ----------------
    // {
    //     std::ofstream csv("fft2d_output.csv");
    //     csv << "u,v,magnitude\n";
    //     for (int row = 0; row < N; ++row) {
    //         double v = freq_axis(row, dy);
    //         for (int col = 0; col < N; ++col) {
    //             double u = freq_axis(col, dx);
    //             csv << u << "," << v << "," << shifted[row * N + col] << "\n";
    //         }
    //     }
    //     std::printf("\nFull shifted magnitude spectrum written to fft2d_output.csv\n");
    // }

    // ---------------- Cleanup host resources ----------------
    free(data.grid_x);
    free(data.grid_y);
    free(data.Gxy);
    free(h_result);

    return 0;
}