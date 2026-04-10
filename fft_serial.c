#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>


double complex* ditfft2(const double *x, size_t N, size_t s, size_t offset){
	double complex* X = (double complex*)malloc(N * sizeof(double complex));

	if (N ==1) {
		X[0] = x[offset];
	}
	else {
		double complex* X_even = (double complex*)malloc(N/2 * sizeof(double complex));
		double complex* X_odd = (double complex*)malloc(N/2 * sizeof(double complex));

		X_even = ditfft2(x, N/2, 2*s, offset);
		X_odd = ditfft2(x, N/2, 2*s, offset+s);

		for (size_t k=0; k < N/2; k++){
			double complex p = X_even[k];
			double complex q = exp(-2*I * M_PI * k / N) * X_odd[k];

			X[k] = p + q;
			X[k + N/2] = p - q;
		}

		free(X_even);
		free(X_odd);
	}

	return X;
}


// define dataset function (1D)

float G(const double x, size_t f){
	return cos(2 * M_PI * f * x);
}


// main function

int main(){
	int N = 32;
	int f = 3;

	FILE* fptr;

	double *grid_x = (double*)malloc(N * sizeof(double));
	double *Gx = (double*)malloc(N * sizeof(double));

	fptr = fopen("Gx_vs_x.json", "w");

	fprintf(fptr, "[\n");

	for(int i=0; i < N; i++){
		grid_x[i] = (double)i / (double)N;
		Gx[i] = G(grid_x[i], f);

		//fprintf(fptr, "%f\t%f\n", grid_x[i], Gx[i]);
		fprintf(fptr, "{\"x\" : %f,\n \"y\" : %f}%s\n", grid_x[i], Gx[i], (i == N-1) ? "" : ",");
	}

	fprintf(fptr, "]\n");

	free(grid_x);
	free(Gx);

	return 0;
}

