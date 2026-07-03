// Function to get elapsed time in seconds 
double get_elapsed_time(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

// define dataset function (2D)
double func_Gxy(const double x, const double y, size_t fx, size_t fy){
	return cos(2 * M_PI * fx * x) * cos(2 * M_PI * fy * y);
}

// define struct to return multiple objects when create dataset
struct dataset{
	double *grid_x;
	double *grid_y;
	double complex *Gxy;
};


// function to create dataset
struct dataset create_dataset(size_t N, size_t fx, size_t fy){
	struct dataset d;

	d.grid_x = (double*)malloc(N * sizeof(double));
	d.grid_y = (double*)malloc(N * sizeof(double));
	d.Gxy = (double complex*)malloc(N * N * sizeof(double complex));

	//create x and y grids
	for(int i=0; i < N; i++){
		d.grid_x[i] = (double)i / (double)N;
		d.grid_y[i] = (double)i / (double)N;
	}

	// create 2D dataset
	for(int i=0; i < N; i++){
		for(int j=0; j < N; j++){
			d.Gxy[i * N + j] = func_Gxy(d.grid_x[i], d.grid_y[j], fx, fy);
		}
	}

	return d;
}

// function to create json file to save dataset (used only for visualization)
void create_json(char* filename, double* x, double* y, double complex* z, size_t N){
	FILE* fptr;
	fptr = fopen(filename, "w");

	fprintf(fptr, "[\n");

	for(int i=0; i < N; i++){
		for(int j=0; j < N; j++){
			fprintf(fptr, "{\"x\" : %f, \"y\" : %f, \"z\" : %f}%s\n", x[i], y[j], creal(z[i * N + j]), (i * N + j == N*N-1) ? "" : ",");
		}
	}

	fprintf(fptr, "]\n");
	fclose(fptr);
}


// ------------------------------------------------------------------ //
// radix-2 Cooley-Tukey algorithm (as following pseudocode)
// ------------------------------------------------------------------ //

// X0,...,N−1 ← ditfft2(x, N, s):             
//     if N = 1 then
//         X0 ← x0                                     trivial size-1 DFT base case
//     else
//         X0,...,N/2−1 ← ditfft2(x, N/2, 2s)             DFT of (x0, x2s, x4s, ..., x(N-2)s)
//         XN/2,...,N−1 ← ditfft2(x+s, N/2, 2s)           DFT of (xs, xs+2s, xs+4s, ..., x(N-1)s)
//         for k = 0 to (N/2)-1 do                      combine DFTs of two halves:
//             p ← Xk
//             q ← exp(−2πi/N k) Xk+N/2
//             Xk ← p + q 
//             Xk+N/2 ← p − q
//         end for
//     end if

double complex* ditfft2(const double complex *x, size_t N, size_t s, size_t offset){		
	double complex* X = (double complex*)malloc(N * sizeof(double complex));	

	if (N ==1) {
		X[0] = x[offset];
	}
	else {
		double complex* X_even = ditfft2(x, N/2, 2*s, offset);
		double complex* X_odd = ditfft2(x, N/2, 2*s, offset+s);

		for (size_t k=0; k < N/2; k++){
			double complex p = X_even[k];
			double complex q = cexp(-2.*I * M_PI * k / (double)N) * X_odd[k];

			X[k] = p + q;
			X[k + N/2] = p - q;
		}

		free(X_even);
		free(X_odd);
	}

	return X;
}

// ************************************************************************************************ //
// the following functions are used to improve the out-of-place radix-2 Cooley-Tukey algorithm
// ************************************************************************************************ //

// ------------------------------------------------------------------ //
//  bit_reverse_copy 
// ------------------------------------------------------------------ //
static void bit_reverse_copy(const double complex *a,
                              double complex *A,
                              size_t n) {
	size_t n_bits = (size_t)log2(n);
    for (size_t k = 0; k < n; k++){
        size_t r = 0;
		size_t tmp = k;
        for (size_t b = 0; b < n_bits; b++) {
            r   = (r << 1) | (tmp & 1);
            tmp >>= 1;
        }
        A[r] = a[k];
    }
}


// ------------------------------------------------------------------------- //
//  Twiddle table - twiddle factors are precomputed to speed up computation
// ------------------------------------------------------------------------- //
static double complex *twiddles  = NULL;
static size_t          twiddle_N = 0;

static void precompute_twiddles(size_t n) {
    if (twiddle_N == n) return;         // avoid to recompute
    free(twiddles);
    twiddles  = (double complex *)malloc((n / 2) * sizeof(double complex));
    twiddle_N = n;
    for (size_t k = 0; k < n / 2; k++){
        twiddles[k] = cexp(-2.0 * I * M_PI * (double)k / (double)n);
	}
}


// ------------------------------------------------------------------ //
// in-place radix-2 Cooley-Tukey algorithm 
// iterative_fft (as following pseudocode, with the difference that
//                 twiddles are precomputed)
// ------------------------------------------------------------------ //

// algorithm iterative-fft is
//     input: Array a of n complex values where n is a power of 2.
//     output: Array A the DFT of a.
 
//     bit-reverse-copy(a, A)
//     n ← a.length 
//     for s = 1 to log(n) do
//         m ← 2s
//         ωm ← exp(−2πi/m) 
//         for k = 0 to n-1 by m do
//             ω ← 1
//             for j = 0 to m/2 – 1 do
//                 t ← ω A[k + j + m/2]
//                 u ← A[k + j]
//                 A[k + j] ← u + t
//                 A[k + j + m/2] ← u – t
//                 ω ← ω ωm
   
//     return A

void iterative_fft(const double complex *a, double complex *A, size_t n) {
 
    precompute_twiddles(n);     // no-op after the first call for size n
 
    bit_reverse_copy(a, A, n);
 
    size_t log_n = (size_t)log2(n);
	
    for (size_t s = 1; s <= log_n; s++) {
        size_t m = (size_t)1 << s;	//fast way to compute 2^s
        size_t step = n / m;
 
        for (size_t k = 0; k < n; k += m) {
            for (size_t j = 0; j < m / 2; j++) {
                double complex w = twiddles[j * step];
                double complex t = w * A[k + j + m / 2];
                double complex u = A[k + j];
                A[k + j]         = u + t;
                A[k + j + m / 2] = u - t;
            }
        }
    }
}
