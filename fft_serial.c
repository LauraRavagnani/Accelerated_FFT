#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>

#include "utils.c"



int main(int argc, char *argv[]){		// put the value of N when run
	bool iterative_mode = true;

	//size_t N = 1024;
	size_t N = atoi(argv[1]);
	size_t fx = 2;
	size_t fy = 2;

	double complex *col = (double complex*)malloc(N * sizeof(double complex));
	double complex *X_kl = (double complex*)malloc(N * N * sizeof(double complex));
	double complex *X_k = (double complex*)malloc(N * sizeof(double complex));
	double complex *X_l = (double complex*)malloc(N * sizeof(double complex));
	double *abs = (double*)malloc(N * N * sizeof(double));
	double *grid_f = (double*)malloc(N * sizeof(double));


	/*  
	* Program main internal variables
	*/
	struct timespec start, t0, t1;
	double elapsed = 0.0;


	// create dataset
	struct dataset data = create_dataset(N, fx, fy);

	// create json with dataset only for N = 128 (used for visualization)
	if(N == 128){
		create_json("Gxy_vs_xy.json", data.grid_x, data.grid_y, data.Gxy, N);
	}

	clock_gettime(CLOCK_MONOTONIC, &start);

	if (iterative_mode == true){

		// perform fft first on rows
		for(int i=0; i < N; i++){
			iterative_fft(data.Gxy + i * N, X_k, N);

			// replace row of original dataset
			for(int j=0; j < N; j++){
				data.Gxy[i * N + j] = X_k[j];
			}
		}

		// extract columns because columns are not contiguous in memory
		for(int i=0; i < N; i++){
			for(int j=0; j < N; j++){
				col[j] = data.Gxy[i + j * N];
			}

			iterative_fft(col, X_l, N);

			for(int j=0; j < N; j++){
				X_kl[i * N + j] = X_l[j];
			}
		}
	} else {
		// perform fft first on rows
		for(int i=0; i < N; i++){
			X_k = ditfft2(data.Gxy, N, 1, i*N);

			// replace row of original dataset
			for(int j=0; j < N; j++){
				data.Gxy[i * N + j] = X_k[j];
			}
		}

		// extract columns because columns are not contiguous in memory
		for(int i=0; i < N; i++){
			for(int j=0; j < N; j++){
				col[j] = data.Gxy[i + j * N];
			}

			X_l = ditfft2(col, N, 1, 0);

			for(int j=0; j < N; j++){
				X_kl[i * N + j] = X_l[j];
			}
		}
	}

	
	clock_gettime(CLOCK_MONOTONIC, &t1);
	elapsed = get_elapsed_time(start, t1);
	printf("Execution time (fft): %.6f seconds\n", elapsed);

	for(int i=0; i < N; i++){
		grid_f[i] = i;
	}

	// create json with fft result
	if(N == 8){
		create_json("fft_prova.json", grid_f, grid_f, X_kl, N);
	}

	free(data.grid_x);
	free(data.grid_y);
	free(grid_f);
	free(data.Gxy);
	free(col);
	free(X_kl);
	free(X_k);
	free(X_l);
	free(abs);



	return 0;
}

