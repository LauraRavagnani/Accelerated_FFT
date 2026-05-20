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


// create dataset
struct dataset create_dataset(size_t N, size_t fx, size_t fy){			//return a pointer
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


// ── bit_reverse_copy ────────────────────────────────────────────────────────
// Each iteration writes to A[r] where r is a unique bit-reversal of k.
// All output indices are distinct → perfectly parallel, no dependencies.
static void bit_reverse_copy(const double complex *a,
                              double complex       *A,
                              size_t                n) {
    size_t n_bits = (size_t)log2(n);

    // Each k maps to a unique r (bit-reversal is a permutation), so
    // every write targets a different A[r]. No race condition possible.
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

// ── precompute_twiddles ─────────────────────────────────────────────────────
// Called once; subsequent calls are a no-op (twiddle_N == n guard).
// The build loop is parallel-safe: each k writes to a unique twiddles[k].
// The guard + malloc must stay sequential — protect with a critical section
// so only one thread allocates, then all threads fill the table in parallel.
static double complex *twiddles  = NULL;
static size_t          twiddle_N = 0;

static void precompute_twiddles(size_t n) {
    // Serialize the check-and-allocate block: if two threads entered
    // simultaneously both would call malloc and one would leak.
    // The omp critical directive identifies a section of code that must be executed by a single thread at a time.
    #pragma omp critical
    {
        if (twiddle_N != n) {
            free(twiddles);
            twiddles  = (double complex *)malloc((n / 2) * sizeof(double complex));
            twiddle_N = n;
        }
    }
    // All threads now see the allocated twiddles pointer.
    // Each k writes to a unique index → safe to parallelise.
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < n / 2; k++)
        twiddles[k] = cexp(-2.0 * I * M_PI * (double)k / (double)n);
}

// ── iterative_fft ───────────────────────────────────────────────────────────
void iterative_fft(const double complex *a, double complex *A, size_t n) {

    precompute_twiddles(n);
    bit_reverse_copy(a, A, n);

    size_t log_n = (size_t)log2(n);

    // s loop is SEQUENTIAL: stage s reads values written by stage s-1.
    // Parallelising across stages would produce wrong results.
    for (size_t s = 1; s <= log_n; s++) {
        size_t m    = (size_t)1 << s;   // 2^s elements per butterfly group
        size_t step = n / m;

        // k loop: each iteration handles a disjoint block A[k .. k+m-1].
        // No two k-iterations share any element → safe to parallelise.
        // The if-guard avoids spawning threads for trivially small stages
        // (e.g. s=1 has only n/2 groups of 2 — overhead outweighs benefit).
        #pragma omp parallel for schedule(static) if(n > 4096)
        for (size_t k = 0; k < n; k += m) {

            // j loop: kept sequential inside each thread.
            // It's only m/2 iterations (at most n/2 at the last stage),
            // and splitting it further would cause false-sharing on
            // adjacent A[k+j] / A[k+j+m/2] pairs.
            for (size_t j = 0; j < m / 2; j++) {
                double complex w = twiddles[j * step];
                double complex t = w * A[k + j + m / 2];
                double complex u = A[k + j];
                A[k + j]         = u + t;
                A[k + j + m / 2] = u - t;
            }
        }
        // Implicit barrier at end of parallel for: all butterflies at
        // stage s are complete before stage s+1 begins.
    }
}