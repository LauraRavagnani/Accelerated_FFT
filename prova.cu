#include <vector>     // For std::vector
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK  2
#define N 8

/* ------------------------------------------------------------------ */
/* Bit-reversal kernel */
/* ------------------------------------------------------------------ */
__global__ void kernel_bit_reverse_copy(const cuDoubleComplex *a,
                                        cuDoubleComplex *A,
                                        unsigned int n,
					                    unsigned int n_bits)
{
    auto k = blockIdx.x * blockDim.x + threadIdx.x; //one element per thread
    if (k < n){
        auto r   = 0;
        auto tmp = k;
        for (auto b = 0; b < n_bits; b++) {
            r   = (r << 1) | (tmp & 1);
            tmp >>= 1;
        }

        A[r] = a[k];
    }
}


int main(int argc, char** argv) {
    size_t n_bits = (size_t)log2(N);

    std::vector<cuDoubleComplex> a(N), A(N);
    
    for (int i = 0; i < N; i++){
       a[i] = make_cuDoubleComplex((double)i, 0.0);
    }


    // device
    cuDoubleComplex* d_a;
    cuDoubleComplex* d_A;

    size_t size = N * sizeof(cuDoubleComplex);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_A, size);

    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);

    printf("\nvector a\n");
    for(int i=0; i<N; i++){
    	printf("a[%d] = (%f, %f)\n", i, cuCreal(a[i]), cuCimag(a[i]));
    }

    int N_b = N / THREADS_PER_BLOCK;
    int N_tpb = THREADS_PER_BLOCK;

    kernel_bit_reverse_copy<<<N_b, N_tpb>>>(d_a, d_A, N, n_bits);

    cudaMemcpy(A.data(), d_A, size, cudaMemcpyDeviceToHost);

    printf("\nvector A\n");
    for(int i=0; i<N; i++){
        printf("A[%d] = (%f, %f)\n", i, cuCreal(A[i]), cuCimag(A[i]));
    }

    cudaFree(d_a);
    cudaFree(d_A);
}

