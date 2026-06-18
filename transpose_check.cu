#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32
#define BLOCK_ROWS 8

#define CUDA_CHECK(call)                                              \
do {                                                                   \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

__global__ void kernel_transpose(double *A_T, const double *A, int width, int height)
{
    __shared__ double tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Read tile from A
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height){
            tile[threadIdx.y + j][threadIdx.x] = A[(y + j) * width + x];
        }
    }

    __syncthreads();

    // Transposed coordinates
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    // Write transposed tile
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width){
            A_T[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void cpu_transpose(double *out,
                   const double *in,
                   int width,
                   int height)
{
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            out[c * height + r] = in[r * width + c];
        }
    }
}

int main()
{
    const int width  = 128;
    const int height = 96;

    size_t bytes = width * height * sizeof(double);

    double *h_A      = (double*)malloc(bytes);
    double *h_A_T    = (double*)malloc(bytes);
    double *h_ref    = (double*)malloc(bytes);

    // Fill matrix with recognizable values
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            h_A[r * width + c] = r * 1000.0 + c;
        }
    }

    cpu_transpose(h_ref, h_A, width, height);

    double *d_A;
    double *d_A_T;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_A_T, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice));

    dim3 block(TILE, BLOCK_ROWS);

    dim3 grid(
        (width  + TILE - 1) / TILE,
        (height + TILE - 1) / TILE);

    kernel_transpose<<<grid, block>>>(
        d_A_T,
        d_A,
        width,
        height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_A_T,
        d_A_T,
        bytes,
        cudaMemcpyDeviceToHost));

    bool passed = true;

    for (int i = 0; i < width * height; i++) {
        if (fabs(h_A_T[i] - h_ref[i]) > 1e-12) {
            printf("Mismatch at %d\n", i);
            printf("GPU = %.15f\n", h_A_T[i]);
            printf("CPU = %.15f\n", h_ref[i]);
            passed = false;
            break;
        }
    }

    if (passed)
        printf("PASS: transpose is correct\n");
    else
        printf("FAIL: transpose is incorrect\n");

    cudaFree(d_A);
    cudaFree(d_A_T);

    free(h_A);
    free(h_A_T);
    free(h_ref);

    return 0;
}