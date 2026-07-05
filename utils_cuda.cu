// ------------------------------------------------------------------------ //
// Dataset helpers                                                          //
// ------------------------------------------------------------------------ //

cuDoubleComplex func_Gxy(const double x, const double y, size_t fx, size_t fy){
    return make_cuDoubleComplex(cos(2.0 * M_PI * fx * x) * cos(2.0 * M_PI * fy * y), 0.0);
}

// define struct to return multiple objects when create dataset
struct dataset {
    double *grid_x;
    double *grid_y;
    cuDoubleComplex *Gxy;
};

// create dataset
struct dataset create_dataset(size_t n, size_t fx, size_t fy)
{
    struct dataset d;
    d.grid_x = (double*)malloc(n * sizeof(double));
    d.grid_y = (double*)malloc(n * sizeof(double));
    d.Gxy = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));

    for (int i = 0; i < (int)n; i++) {
        d.grid_x[i] = (double)i / (double)n;
        d.grid_y[i] = (double)i / (double)n;
    }
    for (int i = 0; i < (int)n; i++)
        for (int j = 0; j < (int)n; j++)
            d.Gxy[i * n + j] = func_Gxy(d.grid_x[i], d.grid_y[j], fx, fy);

    return d;
}

// ------------------------------------------------------------------------ //
// KERNEL in-place bit-reversal                                             //
//                                                                          //    
// Grid  : dim3(N/TPB,  N)                                                  //
// Block : dim3(TPB)                                                        //
//                                                                          //
// Bit-reversal only permutes elements within a row, so rows never          //
// need to interact. Mapping row index onto the grid's y-dimension keeps    //
// that independence explicit                                               //
// ------------------------------------------------------------------------ //
__global__ void kernel_bit_reverse(cuDoubleComplex *A, unsigned int n, unsigned int n_bits){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;

    // compute bit-reversal of tid
    unsigned int r = 0;
    unsigned int tmp = tid;
    for (int b = 0; b < (int)n_bits; b++) {
        r   = (r << 1) | (tmp & 1);
        tmp >>= 1;
    }

    if (r > tid) {           // to avoid double swapping
        cuDoubleComplex t   = A[row * n + tid];
        A[row * n + tid]    = A[row * n + r];
        A[row * n + r]      = t;
    }
}

// ------------------------------------------------------------------------ //
// KERNEL — Cooley-Tukey butterfly                                          //
//                                                                          //
// Grid  : dim3((n/2)/TPB,  n) One thread here touches two array slots      //
//                             (k+j and k+j+half), so n/2 threads are       //
//                             enough to cover an entire row                //
// Block : dim3(TPB)                                                        //
//                                                                          //
// every butterfly pair in a stage is independent of every other pair       //
// in that same stage, so each thread computes one butterfly pair (u, t)    //
// ------------------------------------------------------------------------ //
__global__ void kernel_butterfly(cuDoubleComplex *A, unsigned int n, unsigned int m, unsigned int step){
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;  // tid indexes butterfly pairs, not individual elements
    unsigned int row  = blockIdx.y;                             // which row this block-column is working on
    unsigned int half = m >> 1;                                 // the distance between the two elements one butterfly combines
                                                                // since m is a power of 2, divide m by two means to remove the last 0

    unsigned int k = (tid / half) * m;   // start of the group
    unsigned int j =  tid % half;        // which butterfly within the group

    // twiddle factor  W_n^{j * step} = e^{-2πi * j * step / n}
    double angle = -2.0 * M_PI * (double)(j * step) / (double)n;
    double c, s;
    sincos(angle, &s, &c);
    cuDoubleComplex w = make_cuDoubleComplex(c, s);

    cuDoubleComplex u = A[row * n + k + j];
    cuDoubleComplex t = cuCmul(w, A[row * n + k + j + half]);

    A[row * n + k + j] = cuCadd(u, t);
    A[row * n + k + j + half] = cuCsub(u, make_cuDoubleComplex(cuCreal(t), cuCimag(t)));
}

// ------------------------------------------------------------------------ //
// KERNEL - matrix transpose                                                //
//                                                                          // 
// Grid  : dim3(n/TILE,  n/TILE)                                            //
// Block : dim3(TILE, TILE)                                                 //
// ------------------------------------------------------------------------ //
__global__ void kernel_transpose(const cuDoubleComplex *A, cuDoubleComplex *A_T, unsigned int n)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n)
        A_T[col * n + row] = A[row * n + col];
}

// ---------------------------------------------------------------------- //
// Validation
// four delta peaks of magnitude N^2/4 at (fx, N-fx, fy, N-fy)
// ---------------------------------------------------------------------- //
void validate_fft(const cuDoubleComplex *h, size_t n, size_t fx, size_t fy)
{
    const double tol = 1e-6;
    int errors = 0;
    for (int i = 0; i < (int)n; i++) {
        for (int j = 0; j < (int)n; j++) {
            cuDoubleComplex val = h[i * n + j];
            double diff = 0.0;
            if((i == fx && j == fy) || (i == fx && j == (n - fy)) || (i == (n - fx) && j == fy) || (i == (n - fx) && j == (n - fy))){
                diff = fabs(cuCreal(val) - (double)(n * n) / 4.0);
            }
            else{
                diff = cuCabs(val);
            }

            if (diff > tol) {
                printf("Error at (%d,%d): re=%.6e  im=%.6e\n", i, j, cuCreal(val), cuCimag(val));
                errors++;
            }
        }
    }
    if (errors != 0)
        printf("Validation FAILED: %d error(s).\n", errors);
}


// ------------------------------------------------------------------------ //
// Throughput helpers                                                       //
//   - Compute throughput (GFLOP/s): a 1D FFT of length N does              //
//     log2(N) stages, each with N/2 butterflies; each butterfly is one     //
//     complex multiply (6 flops: 4 mul + 2 add/sub) plus one complex add   //
//     and one complex subtract (4 flops each), i.e. 10 flops/butterfly,    //
//     giving 5*N*log2(N) flops per 1D FFT of length N. A 2D FFT via        //
//     row-column decomposition performs 2*N 1D FFTs (N along rows,         //
//     N along columns), so total flops = 2*N * 5*N*log2(N)                 //
//                                       = 10 * N^2 * log2(N)               //
//   - Memory throughput (GB/s) for the H2D and D2H transfers, computed as  //
//     bytes_transferred / elapsed_time.                                    //
// ------------------------------------------------------------------------ //

static double gflops(unsigned int n, float elapsed_ms){
    double flop_count;
    flop_count = 10.0 * n * n * log2((double)n);
    return ((flop_count / elapsed_ms) / 1e6);
}

static double gbps(size_t bytes, float elapsed_ms){
    return (((double)bytes / elapsed_ms) / 1e6);
}