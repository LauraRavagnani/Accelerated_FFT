#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>

#include "utils.c"



int main(){
	size_t N = 16;
	size_t fx = 2;
	size_t fy = 2;

	double complex *col = (double complex*)malloc(N * sizeof(double complex));
	double complex *X_kl = (double complex*)malloc(N * N * sizeof(double complex));
	double *abs = (double*)malloc(N * N * sizeof(double));
	double *grid_f = (double*)malloc(N * sizeof(double));


	// create dataset
	struct dataset data = create_dataset(N, fx, fy);

	// create json with dataset
	create_json("Gxy_vs_xy.json", data.grid_x, data.grid_y, data.Gxy, N);

	// perform fft first on rows
	for(int i=0; i < N; i++){
		double complex *X_k = ditfft2(data.Gxy, N, 1, i*N);

		// replace row of original dataset
		for(int j=0; j < N; j++){
			data.Gxy[i * N + j] = X_k[j];
		}

		free(X_k);
	}

	// extract columns because columns are not contiguous in memory
	for(int i=0; i < N; i++){
		for(int j=0; j < N; j++){
			col[j] = data.Gxy[i + j * N];
		}

		double complex *X_l = ditfft2(col, N, 1, 0);

		for(int j=0; j < N; j++){
			X_kl[i * N + j] = X_l[j];
		}

		free(X_l);
	}

	for(int i=0; i < N; i++){
		grid_f[i] = i;
	}

	// create json with fft result
	create_json("fft_prova.json", grid_f, grid_f, X_kl, N);


	free(data.grid_x);
	free(data.grid_y);
	free(grid_f);
	free(data.Gxy);
	free(col);
	free(X_kl);
	free(abs);



	return 0;
}

