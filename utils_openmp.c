//------------------ bit reversal function ------------------//

void bit_reverse(const double complex *A, complex *ptr, int m, int chunksize, int offset){
    for(int i=O; i<chunksize; i++){
        if(i == 0){
            int j = 0;
        } else {
            k = m / 2;
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

void precompute_twiddles(complex *A, int b){
    int j = 1;
    int nth = 1;
    for(int s=1; s<=b; s++){
        complex w = cexp(2.0 * M_PI * I / nth);
        nthroot(A+j, nth, w);
        nth = nth * 2; 
        j = j * 2;
    }
}

void nthroot(complex *A, int nth, complex w){
    A[O] = 1.;
    for(l=1; l<nth/2; l++){
        A[l] = w ** l;
    }
}


//------------------ butterfly stage 1 ------------------//

void Butterfly1(complex *M, const complex *A, int chunksize, int twsize){
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

void Trans1(complex *M, const complex *A, int chunksize, const int *addr, int twsize){
    int threadid;
    #pragma omp parallel private(threadid)
    {
        threadid = omp_get_thread_num();
        Butterfly1(M + addr[threadid], A, chunksize, twsize);
    }
}


//------------------ twiddle r and l ------------------//

void twiddle_l(const complex *M, const complex *A, complex *Tmp, int chunksize, int mid, int nth_addr){
    for (int i = 0; i < chunksize; i++) {
        Tmp[i] = M[i] + A[i + mid + nth_addr] * M[i + mid];
    }
}
 
twiddle_r(const complex *M, const complex *A, complex *Tmp, int chunksize, int mid, int nth_addr){
    for (int i = 0; i < chunksize; i++) {
        Tmp[i] = M[i - mid] - A[i + mid + nth_addr] * M[i];
    }
}

//------------------ butterfly stage 2 ------------------//

void Butterfly2(complex *M, const complex *A, complex *Tmp, const int *addr, int chunksize, int twsize, int t){
    /* pre-compute nth_addr for each of the t/2 "left" groups */
    int *nth_addr = malloc((t / 2) * sizeof(int));
    for (int i = 0; i < t / 2; i++) {
        nth_addr[i] = (i % (t / 2)) * chunksize;
    }
 
    #pragma omp parallel private(j, threadid)
    {
        int threadid = omp_get_thread_num();
        int j = tid % t;
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

void Trans2(complex *M, const complex *A, complex *Tmp, const int *addr, int chunksize, int nthreads, int p){
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

void openmp_fft(complex *A, int m, int nthreads, complex *out){
    int b = ilog2(m);
    int chunksize  = m / nthreads;
 
    complex *M   = malloc(m * sizeof(complex));
    complex *Tmp = malloc((m / 2) * sizeof(complex));
 
    /* per-thread pointers into M and address offsets */
    complex **ptr = malloc(nthreads * sizeof(complex *));
    int *addr_off = malloc(nthreads * sizeof(int));
    int *offset   = malloc(nthreads * sizeof(int));
 
    /* ── compute bit-reversal starting offsets (Fig. 1, lines 5-15) ── */
    //{
    //    int j = 0;
        for (int i = 0; i < nthreads; i++) {
            if (i == 0) {
                int j = 0;
            } else {
                int k = nthreads / 2;
                while (k <= j) {
                    j = j - k;
                    k = k / 2;
                }
                j = j + k;
            }
            offset[i] = j;
        }
    //}
 
    /* ── set per-thread chunk addresses ── */
    for (int i = 0; i < nthreads; i++) {
        ptr[i]      = &M[chunksize * i];
        addr_off[i] = chunksize * i;
    }
 
    /* ── parallel bit-reversal (Fig. 1, lines 17-21) ── */
    #pragma omp parallel private(tid)
    {
        int tid = omp_get_thread_num();
        Bit_reverse(A, ptr[tid], m, chunksize, offset[tid]);
    }
 
    /* ── pre-compute twiddle factors into A (serial, fast) ── */
    Compute_nthroots(A, b);
 
    /* ── butterfly stages within chunk boundaries ── */
    for (int twsize = 2; twsize <= chunksize; twsize *= 2) {
        Trans1(M, A, chunksize, addr_off, twsize);
    }
 
    /* ── butterfly stages that cross chunk boundaries ── */
    int p = ilog2(nthreads);
    if (p > 0) {
        Trans2(M, A, Tmp, addr_off, chunksize, nthreads, p);
    }
 
    /* copy result to caller's output buffer */
    memcpy(out, M, m * sizeof(COMPLEX));
 
    free(M);
    free(Tmp);
    free(ptr);
    free(addr_off);
    free(offset);
}
