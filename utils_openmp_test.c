#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include <string.h>

//------------------ bit reversal function ------------------//

void bit_reverse(const double complex *A, double complex *ptr, int m, int chunksize, int offset){
    int j;
    for(int i=0; i<chunksize; i++){
        if(i == 0){
            j = 0;
        } else {
            int k = m / 2;
            while(k <= j){
                j = j - k;
                k = k / 2;
            }
            j = j + k;
        }
        *ptr = A[j + offset];
        ptr++;
    }
}




//------------------ pre-compute twiddles ------------------//

void nthroot(double complex *A, int nth, double complex w){
    A[0] = 1.;
    for(int l=1; l<nth/2; l++){
        A[l] = pow(w, l);
    }
}

void precompute_twiddles(double complex *A, int b){
    int j = 1;
    int nth = 1;
    for(int s=1; s<=b; s++){
        complex w = cexp(2.0 * M_PI * I / nth);
        nthroot(A+j, nth, w);
        nth = nth * 2; 
        j = j * 2;
    }
}




//------------------ butterfly stage 1 ------------------//

void Butterfly1(double complex *M, const double complex *A, int chunksize, int twsize){
    for (int i = 0; i < chunksize; i += twsize) {
        int k = 0;
        for (int j = i; j < i + twsize / 2; j++) {
            complex temp1 = M[j];
            complex temp2 = M[j + twsize/2] * A[k];
            M[j] = temp1 + temp2;
            M[j + twsize/2] = temp1 - temp2;
            k++;
        }
    }
}


//------------------ Trans 1 ------------------//

void Trans1(double complex *M, const double complex *A, int chunksize, const int *addr, int twsize){
    int threadid;
    #pragma omp parallel private(threadid)
    {
        threadid = omp_get_thread_num();
        Butterfly1(M + addr[threadid], A, chunksize, twsize);
    }
}


//------------------ twiddle r and l ------------------//

void twiddle_l(const double complex *M, const double complex *A, double complex *Tmp, int chunksize, int mid, int nth_addr){
    for (int i = 0; i < chunksize; i++) {
        Tmp[i] = M[i] + A[i + mid + nth_addr] * M[i + mid];
    }
}
 
void twiddle_r(const double complex *M, const double complex *A, double complex *Tmp, int chunksize, int mid, int nth_addr){
    for (int i = 0; i < chunksize; i++) {
        Tmp[i] = M[i - mid] - A[i + mid + nth_addr] * M[i];
    }
}

//------------------ butterfly stage 2 ------------------//

void Butterfly2(double complex *M, const double complex *A, double complex *Tmp, const int *addr, int chunksize, int twsize, int t){
    /* pre-compute nth_addr for each of the t/2 "left" groups */
    int *nth_addr = malloc((t / 2) * sizeof(int));
    for (int i = 0; i < t / 2; i++) {
        nth_addr[i] = (i % (t / 2)) * chunksize;
    }

    int j;      // CHECK!!!!!
    int threadid;
    #pragma omp parallel private(j, threadid)
    {
        threadid = omp_get_thread_num();
        j = threadid % t;
        //int mid = twsize / 2;
 
        if (j < t / 2) {
            /* sum ("up arrow") */
            twiddle_l(M + addr[threadid], A, Tmp + addr[threadid], chunksize, twsize/2, nth_addr[j]);
        } else {
            /* difference ("down arrow") */
            twiddle_r(M + addr[threadid], A, Tmp + addr[threadid], chunksize, twsize/2, nth_addr[j-t/2]);
        }
    }
 
    free(nth_addr);
}


//------------------ Trans 2 ------------------//

void Trans2(double complex *M, const double complex *A, double complex *Tmp, const int *addr, int chunksize, int nthreads, int p){
    int twsize = chunksize * 2;
    int t = 2;
 
    for (int i = 0; i < p; i++) {
        if (i % 2 == 0) {
            Butterfly2(M, A, Tmp, addr, chunksize, twsize, t);
        } else {
            Butterfly2(Tmp, A, M, addr, chunksize, twsize, t);
        }
        twsize = twsize * 2;
        t = t * 2;
    }
 
    /* If p is odd the final result ended up in Tmp; copy back to M */
    if (p % 2 == 1) {
        memcpy(M, Tmp, nthreads * chunksize * sizeof(complex));
    }
}


//------------------ openmp_fft ------------------//

void openmp_fft(double complex *A, int m, int nthreads, double complex *out){
    /* Fig. 1, line 2 */
    double complex *M        = malloc(m            * sizeof(double complex));
    double complex *Tmp      = malloc((m / 2)      * sizeof(double complex));
    double complex **ptr     = malloc(nthreads     * sizeof(double complex *));
    int     *addr_off = malloc(nthreads     * sizeof(int));
    int     b        = log2(m);
    int     *offset   = malloc(nthreads     * sizeof(int));
 
    /* Fig. 1, lines 5-15: serial computation of offset[] array.
     * Uses the same divide-and-conquer index walk as Bit_reverse itself
     * to find the starting address of each thread's chunk. */
    int j = 0;
    for (int i = 0; i < nthreads; i++) {
        if (i == 0) {
            j = 0;
        } else {
            int k = nthreads / 2;
            while (k <= j) {
                j -= k;
                k /= 2;
            }
            j += k;
        }
        offset[i] = j;
    }
 
    int      chunksize = m / nthreads;

    /* Fig. 1, lines 17-21: single parallel region.
     * ptr[threadid] is assigned inside the region (line 19),
     * then Bit_reverse is called with that pointer (line 20). */
    int threadid;

    #pragma omp parallel private(threadid)
    {
        threadid = omp_get_thread_num();
 
        /* line 19 */
        ptr[threadid] = &M[chunksize * threadid];
 
        /* line 20 */
        bit_reverse(A, ptr[threadid], m, chunksize, offset[threadid]);
 
        /* addr_off is used by Trans1/Trans2 later; fill it here
         * while each thread knows its own id (no extra loop needed) */
        addr_off[threadid] = chunksize * threadid;
 
    } /* end omp parallel — implicit barrier before next serial phase */
 
    /* Fig. 1, line 22: serial pre-computation of twiddle factors */
    precompute_twiddles(A, b);
 
    /* Fig. 1, lines 23-24: Trans1 — stages within chunk boundary */
    for (int twsize = 2; twsize <= chunksize; twsize *= 2){
        Trans1(M, A, chunksize, addr_off, twsize);
    }

 
    /* Fig. 1, line 25: Trans2 — stages that cross chunk boundary */
    int p = (int)log2(nthreads);
    // if (p > 0)
        Trans2(M, A, Tmp, addr_off, chunksize, nthreads, p);
 
    /* copy result out and release working memory */
    memcpy(out, M, m * sizeof(double complex));
 
    free(M);
    free(Tmp);
    free(ptr);
    free(addr_off);
    free(offset);
}
 
int main(void)
{
    const int m        = 16;    /* must be power of 2 */
    const int nthreads = 4;
    const int k0       = 2;     /* frequency bin of the test cosine */
 
    omp_set_num_threads(nthreads);
 
    double complex *A   = malloc(m * sizeof(double complex));
    double complex *out = malloc(m * sizeof(double complex));
 
    /* input: cos(2π k0 n / m) */
    for (int n = 0; n < m; n++) {
        A[n] = cos(2 * M_PI * k0 * n/m);
    }

 
    double t0 = omp_get_wtime();
    openmp_fft(A, m, nthreads, out);
    double t1 = omp_get_wtime();
 
    printf("\nFFT output magnitudes (expect peaks at bins %d and %d):\n",
           k0, m - k0);
    for (int k = 0; k < m; k++) {
        double mag = cabs(out[k]);
        printf("  |X[%2d]| = %7.4f\n", k, mag);
    }
 
    printf("\nElapsed: %.6f s  (m=%d, threads=%d)\n", t1 - t0, m, nthreads);
 
    free(A);
    free(out);
    return 0;
}
