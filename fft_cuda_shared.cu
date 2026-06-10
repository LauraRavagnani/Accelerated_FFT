// Precomputed twiddle factors in constant memory
__constant__ float2 c_twiddles[2048];

void init_twiddles(int max_fft_size) {
    float2* h_twiddles = new float2[max_fft_size / 2];
    for (int k = 0; k < max_fft_size / 2; k++) {
        float angle = -2.0f * M_PI * k / max_fft_size;
        h_twiddles[k].x = cosf(angle);
        h_twiddles[k].y = sinf(angle);
    }
    cudaMemcpyToSymbol(c_twiddles, h_twiddles,
                       (max_fft_size / 2) * sizeof(float2));
    delete[] h_twiddles;
}

__global__ void fft_stage_optimized(float2* data, int stage, int fft_size) {
    __shared__ float2 s_data[1024 + 32];  // Padded for bank conflicts

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    // Coalesced load with padding
    if (idx < fft_size) {
        int padded_tid = tid + tid / 32;
        s_data[padded_tid] = data[idx];
    }
    __syncthreads();

    int stride = 1 << stage;
    int num_butterflies_per_block = blockDim.x / (2 * stride);

    for (int iter = 0; iter < (blockDim.x / (2 * stride)); iter++) {
        int butterfly_idx = tid / (2 * stride);
        int pos_in_butterfly = tid % stride;
        int offset = butterfly_idx * (2 * stride);

        int i = offset + pos_in_butterfly;
        int j = i + stride;

        if (i < blockDim.x && j < blockDim.x) {
            // Lookup precomputed twiddle
            int twiddle_idx = pos_in_butterfly * (fft_size / (2 * stride));
            float2 twiddle = c_twiddles[twiddle_idx];

            int padded_i = i + i / 32;
            int padded_j = j + j / 32;

            float2 u = s_data[padded_i];
            float2 v = s_data[padded_j];

            // Complex multiply: v * twiddle
            float2 v_t;
            v_t.x = v.x * twiddle.x - v.y * twiddle.y;
            v_t.y = v.x * twiddle.y + v.y * twiddle.x;

            // Butterfly
            s_data[padded_i] = make_float2(u.x + v_t.x, u.y + v_t.y);
            s_data[padded_j] = make_float2(u.x - v_t.x, u.y - v_t.y);
        }
        __syncthreads();
    }

    // Coalesced write
    if (idx < fft_size) {
        int padded_tid = tid + tid / 32;
        data[idx] = s_data[padded_tid];
    }
}

void compute_fft_optimized(float2* d_data, int fft_size) {
    int log2n = __builtin_ctz(fft_size);  // Fast log2

    // Initialize twiddles once
    static bool twiddles_initialized = false;
    if (!twiddles_initialized) {
        init_twiddles(fft_size);
        twiddles_initialized = true;
    }

    // Bit-reversal (can be fused with stage 0)
    int threads = 256;
    int blocks = (fft_size + threads - 1) / threads;
    // ... bit_reverse_kernel<<<blocks, threads>>>(d_data, fft_size);

    // Execute FFT stages
    for (int stage = 0; stage < log2n; stage++) {
        fft_stage_optimized<<<blocks, threads>>>(d_data, stage, fft_size);
    }
}

