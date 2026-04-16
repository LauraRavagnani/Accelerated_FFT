// radix-2 Cooley-Tukey algorithm
double complex* ditfft2(const double complex *x, size_t N, size_t s, size_t offset){		//return a pointer
	double complex* X = (double complex*)malloc(N * sizeof(double complex));	// don't use malloc inside function!!!

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