#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// Dataset helpers
// ============================================================================

cuDoubleComplex func_Gxy(const double x,
                         const double y,
                         size_t fx,
                         size_t fy)
{
    return make_cuDoubleComplex(
        cos(2.0 * M_PI * fx * x) *
        cos(2.0 * M_PI * fy * y),
        0.0
    );
}

struct dataset
{
    double* grid_x;
    double* grid_y;
    cuDoubleComplex* Gxy;
};

struct dataset create_dataset(size_t n,
                              size_t fx,
                              size_t fy)
{
    struct dataset d;

    d.grid_x =
        (double*)malloc(n*sizeof(double));

    d.grid_y =
        (double*)malloc(n*sizeof(double));

    d.Gxy =
        (cuDoubleComplex*)
        malloc(n*n*sizeof(cuDoubleComplex));

    for(int i=0;i<(int)n;i++)
    {
        d.grid_x[i]=(double)i/(double)n;
        d.grid_y[i]=(double)i/(double)n;
    }

    for(int i=0;i<(int)n;i++)
    {
        for(int j=0;j<(int)n;j++)
        {
            d.Gxy[i*n+j] =
                func_Gxy(
                    d.grid_x[i],
                    d.grid_y[j],
                    fx,
                    fy
                );
        }
    }

    return d;
}

// ============================================================================
// Bit reverse helper
// ============================================================================

__device__ __forceinline__
unsigned int bit_reverse(unsigned int x,
                         unsigned int n_bits)
{
    unsigned int r = 0;

    for(unsigned int i=0;i<n_bits;i++)
    {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }

    return r;
}

// ============================================================================
// Shared-memory FFT kernel
//
// One block = one complete row
// 256 threads for N=512
// Entire row stored in shared memory
// ============================================================================

__global__
void kernel_fft_smem(cuDoubleComplex *A,
                     unsigned int n,
                     unsigned int n_bits)
{
    extern __shared__ cuDoubleComplex smem[];

    const unsigned int tid =
        threadIdx.x;

    const unsigned int row =
        blockIdx.x;

    cuDoubleComplex *row_ptr =
        A + row*n;

    // --------------------------------------------------------------------
    // Load row
    // --------------------------------------------------------------------

    smem[tid] = row_ptr[tid];

    smem[tid+n/2] = row_ptr[tid+n/2];

    __syncthreads();

    // --------------------------------------------------------------------
    // Bit reversal
    // --------------------------------------------------------------------

    for(int pass=0; pass<2; pass++)
    {
        unsigned int k =
            tid + pass*(n/2);

        unsigned int r =
            bit_reverse(k,n_bits);

        if(r > k)
        {
            cuDoubleComplex tmp =
                smem[k];

            smem[k] =
                smem[r];

            smem[r] =
                tmp;
        }
    }

    __syncthreads();

    // --------------------------------------------------------------------
    // FFT stages
    // --------------------------------------------------------------------

    for(unsigned int stage=1;
                     stage<=n_bits;
                     stage++)
    {
        unsigned int m =
            1u << stage;

        unsigned int half =
            m >> 1;

        unsigned int step =
            n / m;

        unsigned int k =
            (tid / half) * m;

        unsigned int j =
            tid % half;

        double angle =
            -2.0 * M_PI *
            (double)(j * step) /
            (double)n;

        double c,s;

        sincos(angle,&s,&c);

        cuDoubleComplex w =
            make_cuDoubleComplex(c,s);

        cuDoubleComplex u =
            smem[k+j];

        cuDoubleComplex t =
            cuCmul(
                w,
                smem[k+j+half]
            );

        __syncthreads();

        smem[k+j] =
            cuCadd(u,t);

        smem[k+j+half] =
            cuCsub(u,t);

        __syncthreads();
    }

    // --------------------------------------------------------------------
    // Store row
    // --------------------------------------------------------------------

    row_ptr[tid] =
        smem[tid];

    row_ptr[tid+n/2] =
        smem[tid+n/2];
}

// ============================================================================
// Transpose kernel
// ============================================================================

__global__
void kernel_transpose(const cuDoubleComplex *A,
                            cuDoubleComplex *AT,
                            unsigned int n)
{
    unsigned int col =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    unsigned int row =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if(row < n && col < n)
    {
        AT[col*n + row] =
            A[row*n + col];
    }
}

// ============================================================================
// FFT wrapper
// ============================================================================

void fft_rows(cuDoubleComplex *d_mat,
              unsigned int n,
              unsigned int n_bits)
{
    dim3 block(n/2);     // 256 threads
    dim3 grid(n);        // one block per row

    size_t smem_bytes =
        n*sizeof(cuDoubleComplex);

    kernel_fft_smem<<<
        grid,
        block,
        smem_bytes
    >>>(
        d_mat,
        n,
        n_bits
    );

    cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------
// Validation helper
// -----------------------------------------------------------------------------
void validate_fft(const cuDoubleComplex *h, size_t n, size_t fx, size_t fy)
{
    const double tol = 1e-2;
    int errors = 0;
    for (int i = 0; i < (int)n; i++) {
        for (int j = 0; j < (int)n; j++) {
            cuDoubleComplex val = h[i * n + j];
            double diff;
            bool is_peak = (i == (int)fx       && j == (int)fy)
                        || (i == (int)fx       && j == (int)(n - fy))
                        || (i == (int)(n - fx) && j == (int)fy)
                        || (i == (int)(n - fx) && j == (int)(n - fy));
            if (is_peak)
                diff = fabs(cuCreal(val) - (double)(n * n) / 4.0);
            else
                diff = cuCabs(val);

            if (diff > tol) {
                printf("Error at (%d,%d): re=%.6e  im=%.6e\n",
                       i, j, cuCreal(val), cuCimag(val));
                errors++;
            }
        }
    }
    if (errors != 0)
        printf("Validation FAILED: %d error(s).\n", errors);
}
