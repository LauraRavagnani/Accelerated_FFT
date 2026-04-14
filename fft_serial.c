#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>

// radix-2 Cooley-Tukey algorithm
double complex* ditfft2(const double complex *x, size_t N, size_t s, size_t offset){
	double complex* X = (double complex*)malloc(N * sizeof(double complex));	// don't use malloc inside function!!!

	if (N ==1) {
		X[0] = x[offset];
	}
	else {
		// double complex* X_even = (double complex*)malloc(N/2 * sizeof(double complex));
		// double complex* X_odd = (double complex*)malloc(N/2 * sizeof(double complex));

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


// define dataset function (1D)

double func_Gx(const double x, size_t f){
	return cos(2 * M_PI * f * x);
}

// define dataset function (2D)
double func_Gxy(const double x, const double y, size_t fx, size_t fy){
	return cos(2 * M_PI * fx * x) * cos(2 * M_PI * fy * y);
}

//functio to write results into json files
// void write_json(filename){
// 	FILE* fptr;

// 	fptr = fopen(filename, "w");

// 	fprintf(fptr, "[\n");
// }

// main function

int main(){
	size_t N = 16;
	size_t fx = 2;
	size_t fy = 2;


	//FILE* fptr;
	// FILE* fptr_xy;
	FILE* fptr_fft;

	double *grid_x = (double*)malloc(N * sizeof(double));
	double *grid_y = (double*)malloc(N * sizeof(double));
	// //double *Gx = (double*)malloc(N * sizeof(double));
	double complex *Gxy = (double complex*)malloc(N * N * sizeof(double complex));
	double complex *col = (double complex*)malloc(N * sizeof(double complex));
	double complex *X_kl = (double complex*)malloc(N * N * sizeof(double complex));
	double *abs = (double*)malloc(N * N * sizeof(double));

	// fptr = fopen("Gx_vs_x.json", "w");
	// fptr_xy = fopen("Gxy_vs_xy.json", "w");
	fptr_fft = fopen("fft.json", "w");

	// fprintf(fptr, "[\n");
	// fprintf(fptr_xy, "[\n");
	fprintf(fptr_fft, "[\n");

	// create x and y grid
	for(int i=0; i < N; i++){
		grid_x[i] = (double)i / (double)N;
		grid_y[i] = (double)i / (double)N;
	}


	// create 2D dataset
	for(int i=0; i < N; i++){
		for(int j=0; j < N; j++){
			Gxy[i * N + j] = func_Gxy(grid_x[i], grid_y[j], fx, fy);

			//fprintf(fptr_xy, "{\"x\" : %f, \"y\" : %f, \"Gxy\" : %f}%s\n", grid_x[i], grid_y[j], Gxy[i * N + j], (i * N + j == N*N-1) ? "" : ",");
		}
	}


	// perform fft first on rows and then on columns
	for(int i=0; i < N; i++){
		double complex *X_k = ditfft2(Gxy, N, 1, i*N);

		for(int j=0; j < N; j++){
			Gxy[i * N + j] = X_k[j];
		}

		free(X_k);
	}

	//extract columns because columns are not contiguous in memory
	for(int i=0; i < N; i++){
		for(int j=0; j < N; j++){
			col[j] = Gxy[i + j * N];
		}

		double complex *X_l = ditfft2(col, N, 1, 0);

		for(int j=0; j < N; j++){
			X_kl[i * N + j] = X_l[j];

			fprintf(fptr_fft, "{\"x\" : %f, \"y\" : %f, \"mag\" : %f}%s\n", grid_x[i], grid_y[j], cabs(X_kl[i * N + j]), (i * N + j == N*N-1) ? "" : ",");
		}

		free(X_l);
	}




	//double complex *X_k = ditfft2(Gx, N, 1, 0);

	// for(int i=0; i < N; i++){
	// 	abs[i] = cabs(X_k[i]);

	// 	//create json file for plot
	// 	fprintf(fptr, "{\"x\" : %f, \"y\" : %f, \"abs\" : %f}%s\n", grid_x[i], Gx[i], abs[i], (i == N-1) ? "" : ",");
	// }

	// fprintf(fptr, "]\n");
	// fclose(fptr);

	// fprintf(fptr_xy, "]\n");
	// fclose(fptr_xy);

	fprintf(fptr_fft, "]\n");
	fclose(fptr_fft);


	free(grid_x);
	free(grid_y);
	free(Gxy);
	free(col);
	free(X_kl);
	free(abs);


	return 0;
}

