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


// create dataset
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

// function to create json file to save fft result (used only for visualization)
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

// Function to get elapsed time in ms 
double get_elapsed_time(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

// ************************************************************************************************ //
// the following functions are used to implement the in-place radix-2 Cooley-Tukey algorithm
// since it is the fastest compared to the out-of-place
// ************************************************************************************************ //

// ------------------------------------------------------------------ //
//  bit_reverse_copy 
// ------------------------------------------------------------------ //

// Each iteration writes to A[r] where r is a unique bit-reversal of k.
// All output indices are distinct → perfectly parallel, no dependencies.
static void bit_reverse_copy(const double complex *a, double complex *A, size_t n) {
    size_t n_bits = (size_t)log2(n);

    // schedule(static) divides the iteration space evenly before execution,
    // minimising scheduling overhead because every iteration performs roughly
    // the same amount of work.
    //
    // Each iteration computes the bit-reversed index for one element and writes
    // to a unique location A[r]. Since bit reversal is a permutation, it's impossible
    // for two threads to write to the same array element => no race conditions
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < n; k++) {
        size_t r   = 0;
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

// The build loop is parallel-safe: each k writes to a unique twiddles[k].
// The guard + malloc must stay sequential — protect with a critical section
// so only one thread allocates, then all threads fill the table in parallel

static double complex *twiddles  = NULL;
static size_t          twiddle_N = 0;

static void precompute_twiddles(size_t n) {
    // The critical directive serialises this section while leaving the rest of
    // the function parallel
    #pragma omp critical
    {
        if (twiddle_N != n) {
            free(twiddles);
            twiddles  = (double complex *)malloc((n / 2) * sizeof(double complex));
            twiddle_N = n;
        }
    }
    // Every thread writes to a different twiddles[k]
    // Static scheduling provides the lowest overhead because all
    // iterations require identical work
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < n / 2; k++)
        twiddles[k] = cexp(-2.0 * I * M_PI * (double)k / (double)n);
}

// ------------------------------------------------------------------ //
// in-place radix-2 Cooley-Tukey algorithm 
// iterative_fft
// ------------------------------------------------------------------ //
void iterative_fft(const double complex *a, double complex *A, size_t n) {

    precompute_twiddles(n);
    bit_reverse_copy(a, A, n);

    size_t log_n = (size_t)log2(n);

    // s loop is sequential: stage s reads values written by stage s-1
    // Parallelising across stages would produce wrong results
    for (size_t s = 1; s <= log_n; s++) {
        size_t m    = (size_t)1 << s;   // 2^s elements per butterfly group
        size_t step = n / m;

        // k loop: each iteration handles a disjoint block of A
        // No two k-iterations share any element => safe to parallelise
        // #pragma omp parallel for schedule(static) //if(n > 4096)
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


// double complex* fftshift_2d(const double complex *X, size_t N) {
   
//     double complex *X_shift = (double complex *)malloc(N * N * sizeof(double complex));

//     #pragma omp parallel for schedule(static) collapse(2)
//     for (size_t i = 0; i < N/2; i++) {
//         for (size_t j = 0; j < N/2; j++) {

//             // Q0 (top-left)     -> center of Q3 (bottom-right)
//             X_shift[(i + N/2) * N + (j + N/2)] = X[i * N + j];

//             // Q3 (bottom-right) -> center of Q0 (top-left)
//             X_shift[i * N + j]                   = X[(i + N/2) * N + (j + N/2)];

//             // Q1 (top-right)    -> center of Q2 (bottom-left)
//             X_shift[(i + N/2) * N + j]          = X[i * N + (j + N/2)];

//             // Q2 (bottom-left)  -> center of Q1 (top-right)
//             X_shift[i * N + (j + N/2)]          = X[(i + N/2) * N + j];
//         }
//     }

//     return X_shift;
// }


// ------------------------------------------------------------------------- //
// Validate the implementation against the expected cosine transform output  //
// ------------------------------------------------------------------------- //

void validate_fft(double complex *X, size_t N, size_t fx, size_t fy) {
    double tolerance = 1e-6;
    int n_errors = 0;

    // collapse(2) Flatten the nested loops into one iteration space and distribute the
    // iterations across threads
    // reduction(+:n_errors) gives every thread a private copy of n_errors
    // After the loop finishes, OpenMP automatically sums the private values
    // into the shared variable
    #pragma omp parallel for schedule(static) collapse(2) reduction(+:n_errors)
    for(int i=0; i<(int)N; i++){
        for(int j=0; j<(int)N; j++){
            double diff = 0.0;
            if((i == fx && j == fy) || (i == fx && j == N-fy) || (i == N-fx && j == fy) || (i == N-fx && j == N-fy)){
                diff = fabs(X[i * N + j] - pow(N, 2)/4);
            } else {
                diff = X[i * N + j];        
            }

            if(diff > tolerance){
                // Prevent multiple threads from printing simultaneously
                #pragma omp critical
                printf("Failed verification");
                n_errors += 1;
            };
        }
    }
}