// =============================================================================
// image_fft_filter.cu
//
// Frequency-domain image filtering built on top of the same kernels as
// fft_cuda_global.cu / ifft_cuda_global.cu:
//
//   kernel_bit_reverse, kernel_butterfly            (forward FFT)
//   kernel_bit_reverse, kernel_butterfly_inverse     (inverse FFT)
//   kernel_transpose                                 (shared by both)
//   kernel_scale                                     (1/N^2 IFFT normalization)
//
// New kernels added, specific to image processing:
//   kernel_load_real     : copy uint8 pixels into the real part of a padded
//                           complex buffer (imag = 0), zero-pad the rest
//   kernel_fftshift       : swap quadrants so DC sits at the image center
//                           (standard for visualizing/filtering spectra)
//   kernel_apply_lowpass  : multiply spectrum by a Gaussian low-pass mask
//                           (swap this kernel out for high-pass / band-pass /
//                           sharpen / edge-detect masks)
//   kernel_to_real_u8     : take the real part, clamp to [0,255], write uint8
//
// Pipeline:
//   image (uint8, W x H)
//     -> pad to N x N (N = next pow2 of max(W,H))
//     -> kernel_load_real                (real image  -> complex buffer)
//     -> fft_rows -> transpose -> fft_rows        (2D forward FFT)
//     -> kernel_fftshift                  (center the spectrum)
//     -> kernel_apply_lowpass             (frequency-domain filter)
//     -> kernel_fftshift                  (undo centering, back to FFT order)
//     -> ifft_rows -> transpose -> ifft_rows -> kernel_scale   (2D inverse FFT)
//     -> kernel_to_real_u8                (complex -> uint8)
//     -> crop back to W x H, write PNG
//
// Compile:
//   nvcc -O2 -arch=sm_75 -o image_fft_filter.out image_fft_filter.cu
//
// Usage:
//   ./image_fft_filter.out input.png output.png <cutoff_fraction>
//     cutoff_fraction: 0.0-1.0, fraction of spectrum radius to keep
//                       (e.g. 0.15 = strong blur, 0.4 = mild blur)
// =============================================================================

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// -----------------------------------------------------------------------------
// Shared kernels (identical to fft_cuda_global.cu / ifft_cuda_global.cu)
// -----------------------------------------------------------------------------

__global__ void kernel_bit_reverse(cuDoubleComplex *A,
                                   unsigned int     n,
                                   unsigned int     n_bits)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (tid >= n) return;

    unsigned int r = 0;
    unsigned int tmp = tid;
    for (int b = 0; b < (int)n_bits; b++) {
        r   = (r << 1) | (tmp & 1);
        tmp >>= 1;
    }
    if (r > tid) {
        cuDoubleComplex t = A[row * n + tid];
        A[row * n + tid]  = A[row * n + r];
        A[row * n + r]    = t;
    }
}

__global__ void kernel_butterfly(cuDoubleComplex *A, unsigned int n,
                                 unsigned int m, unsigned int step)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row  = blockIdx.y;
    unsigned int half = m >> 1;
    if (tid >= n / 2) return;

    unsigned int k = (tid / half) * m;
    unsigned int j =  tid % half;

    double angle = -2.0 * M_PI * (double)(j * step) / (double)n;
    double c, s;
    sincos(angle, &s, &c);
    cuDoubleComplex w = make_cuDoubleComplex(c, s);

    cuDoubleComplex u = A[row * n + k + j];
    cuDoubleComplex t = cuCmul(w, A[row * n + k + j + half]);
    A[row * n + k + j]        = cuCadd(u, t);
    A[row * n + k + j + half] = cuCsub(u, t);
}

__global__ void kernel_butterfly_inverse(cuDoubleComplex *A, unsigned int n,
                                         unsigned int m, unsigned int step)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row  = blockIdx.y;
    unsigned int half = m >> 1;
    if (tid >= n / 2) return;

    unsigned int k = (tid / half) * m;
    unsigned int j =  tid % half;

    double angle = 2.0 * M_PI * (double)(j * step) / (double)n;
    double c, s;
    sincos(angle, &s, &c);
    cuDoubleComplex w = make_cuDoubleComplex(c, s);

    cuDoubleComplex u = A[row * n + k + j];
    cuDoubleComplex t = cuCmul(w, A[row * n + k + j + half]);
    A[row * n + k + j]        = cuCadd(u, t);
    A[row * n + k + j + half] = cuCsub(u, t);
}

__global__ void kernel_transpose(const cuDoubleComplex *A,
                                       cuDoubleComplex *A_T,
                                       unsigned int     n)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n)
        A_T[col * n + row] = A[row * n + col];
}

__global__ void kernel_scale(cuDoubleComplex *A, unsigned int n)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (tid >= n) return;
    double inv_n2 = 1.0 / ((double)n * (double)n);
    cuDoubleComplex v = A[row * n + tid];
    A[row * n + tid] = make_cuDoubleComplex(cuCreal(v) * inv_n2, cuCimag(v) * inv_n2);
}

// -----------------------------------------------------------------------------
// Image-specific kernels
// -----------------------------------------------------------------------------

// Copy uint8 pixels (W x H, row-major) into the real part of an N x N
// complex buffer, zero everywhere outside the W x H region (zero-padding).
__global__ void kernel_load_real(const unsigned char *img, int W, int H,
                                 cuDoubleComplex *A, unsigned int n)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (col >= n) return;

    double v = 0.0;
    if (row < (unsigned int)H && col < (unsigned int)W)
        v = (double)img[row * W + col];

    A[row * n + col] = make_cuDoubleComplex(v, 0.0);
}

// Swap quadrants (top-left <-> bottom-right, top-right <-> bottom-left) so
// the zero-frequency (DC) component moves from the corner to the center.
// Only touches the top half; each thread swaps one pixel with its
// diagonally-opposite quadrant partner. n must be even (true for any pow2).
__global__ void kernel_fftshift(cuDoubleComplex *A, unsigned int n)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    unsigned int half = n / 2;
    if (col >= half || row >= half) return;

    unsigned int r2 = row + half, c2 = col + half;

    cuDoubleComplex tmp1 = A[row * n + col];
    A[row * n + col]     = A[r2 * n + c2];
    A[r2 * n + c2]        = tmp1;

    cuDoubleComplex tmp2 = A[row * n + c2];
    A[row * n + c2]      = A[r2 * n + col];
    A[r2 * n + col]      = tmp2;
}

// Gaussian low-pass mask centered on a shifted spectrum: attenuates
// frequencies farther than `cutoff` (in pixels) from the center.
// Swap this kernel's body for other filters, e.g.:
//   - high-pass: multiply by (1 - gaussian) instead
//   - ideal circular low/high-pass: hard cutoff instead of exp()
//   - band-pass: subtract two Gaussians of different radius
__global__ void kernel_apply_lowpass(cuDoubleComplex *A, unsigned int n,
                                     double cutoff)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (col >= n) return;

    double cx = n / 2.0, cy = n / 2.0;
    double dx = (double)col - cx, dy = (double)row - cy;
    double dist2 = dx * dx + dy * dy;

    double mask = (1. - exp(-dist2 / (2.0 * cutoff * cutoff)));

    cuDoubleComplex v = A[row * n + col];
    A[row * n + col] = make_cuDoubleComplex(cuCreal(v) * mask, cuCimag(v) * mask);
}

// Take the real part, clamp to [0,255], write back as uint8 (cropped to W x H).
__global__ void kernel_to_real_u8(const cuDoubleComplex *A, unsigned int n,
                                  unsigned char *out, int W, int H)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (col >= (unsigned int)W || row >= (unsigned int)H) return;

    double v = cuCreal(A[row * n + col]);
    if (v < 0.0) v = 0.0;
    if (v > 255.0) v = 255.0;
    out[row * W + col] = (unsigned char)(v + 0.5);
}

// -----------------------------------------------------------------------------
// Row-pass helpers (identical structure to fft_rows / ifft_rows)
// -----------------------------------------------------------------------------

static void fft_rows(cuDoubleComplex *d_mat, unsigned int n, unsigned int n_bits)
{
    int TPB = 256;
    dim3 g_br(n / TPB, n);
    kernel_bit_reverse<<<g_br, TPB>>>(d_mat, n, n_bits);
    cudaDeviceSynchronize();

    dim3 g_bf((n / 2) / TPB, n);
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m = 1u << s, step = n / m;
        kernel_butterfly<<<g_bf, TPB>>>(d_mat, n, m, step);
        cudaDeviceSynchronize();
    }
}

static void ifft_rows(cuDoubleComplex *d_mat, unsigned int n, unsigned int n_bits)
{
    int TPB = 256;
    dim3 g_br(n / TPB, n);
    kernel_bit_reverse<<<g_br, TPB>>>(d_mat, n, n_bits);
    cudaDeviceSynchronize();

    dim3 g_bf((n / 2) / TPB, n);
    for (unsigned int s = 1; s <= n_bits; s++) {
        unsigned int m = 1u << s, step = n / m;
        kernel_butterfly_inverse<<<g_bf, TPB>>>(d_mat, n, m, step);
        cudaDeviceSynchronize();
    }
}

static unsigned int next_pow2(unsigned int x)
{
    unsigned int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input> <output> [cutoff_fraction=0.2]\n", argv[0]);
        return 1;
    }
    const char *in_path  = argv[1];
    const char *out_path = argv[2];
    double cutoff_frac = (argc >= 4) ? atof(argv[3]) : 0.2;

    int W, H, channels;
    unsigned char *img = stbi_load(in_path, &W, &H, &channels, 1); // force grayscale
    if (!img) {
        fprintf(stderr, "Failed to load %s\n", in_path);
        return 1;
    }
    printf("Loaded %s: %d x %d (grayscale)\n", in_path, W, H);

    unsigned int n = next_pow2((unsigned int)((W > H) ? W : H));
    unsigned int n_bits = (unsigned int)log2((double)n);
    double cutoff = cutoff_frac * (n / 2.0);
    printf("Padded to %u x %u, cutoff radius = %.1f px\n", n, n, cutoff);

    size_t csize = (size_t)n * n * sizeof(cuDoubleComplex);

    unsigned char *d_img_in, *d_img_out;
    cuDoubleComplex *d_A, *d_A_T;
    cudaMalloc(&d_img_in,  (size_t)W * H);
    cudaMalloc(&d_img_out, (size_t)W * H);
    cudaMalloc(&d_A,   csize);
    cudaMalloc(&d_A_T, csize);
    cudaMemcpy(d_img_in, img, (size_t)W * H, cudaMemcpyHostToDevice);

    int TPB = 256;

    // load + zero-pad
    dim3 g_load(n / TPB, n);
    kernel_load_real<<<g_load, TPB>>>(d_img_in, W, H, d_A, n);
    cudaDeviceSynchronize();

    // forward 2D FFT
    fft_rows(d_A, n, n_bits);
    kernel_transpose<<<dim3(n / 32, n / 32), dim3(32, 32)>>>(d_A, d_A_T, n);
    cudaDeviceSynchronize();
    fft_rows(d_A_T, n, n_bits);

    // center spectrum, filter, un-center
    // fftshift grid must cover n/2 x n/2 threads
    dim3 g_shift2((n / 2 + TPB - 1) / TPB, n / 2);
    kernel_fftshift<<<g_shift2, TPB>>>(d_A_T, n);
    cudaDeviceSynchronize();

    dim3 g_filter(n / TPB, n);
    kernel_apply_lowpass<<<g_filter, TPB>>>(d_A_T, n, cutoff);
    cudaDeviceSynchronize();

    kernel_fftshift<<<g_shift2, TPB>>>(d_A_T, n); // shift back
    cudaDeviceSynchronize();

    // inverse 2D FFT
    ifft_rows(d_A_T, n, n_bits);
    kernel_transpose<<<dim3(n / 32, n / 32), dim3(32, 32)>>>(d_A_T, d_A, n);
    cudaDeviceSynchronize();
    ifft_rows(d_A, n, n_bits);
    kernel_scale<<<dim3(n / TPB, n), TPB>>>(d_A, n);
    cudaDeviceSynchronize();

    // crop back to W x H, clamp, write out
    dim3 g_out((W + TPB - 1) / TPB, H);
    kernel_to_real_u8<<<g_out, TPB>>>(d_A, n, d_img_out, W, H);
    cudaDeviceSynchronize();

    unsigned char *out = (unsigned char*)malloc((size_t)W * H);
    cudaMemcpy(out, d_img_out, (size_t)W * H, cudaMemcpyDeviceToHost);

    stbi_write_png(out_path, W, H, 1, out, W);
    printf("Wrote %s\n", out_path);

    stbi_image_free(img);
    free(out);
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(d_A);
    cudaFree(d_A_T);
    return 0;
}
