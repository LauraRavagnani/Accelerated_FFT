# Accelerated FFT

High-performance implementations of the **2D Fast Fourier Transform (FFT)** based on the **Radix-2 Cooley–Tukey algorithm**, progressing from a sequential CPU implementation to parallel CPU and GPU accelerations.

The project compares multiple optimization strategies, including:

- Sequential C implementation
- OpenMP multi-threading
- CUDA implementation
- CUDA implementation using shared memory
- NVIDIA cuFFT library
- Performance benchmarking and scaling analysis

---

## Overview

The Fast Fourier Transform (FFT) is one of the most important algorithms in scientific computing, enabling efficient computation of the Discrete Fourier Transform (DFT) with a complexity of

\[
O(N \log N)
\]

instead of

\[
O(N^2)
\]

This project implements a **2D FFT** using the **Radix-2 Decimation-in-Time (DIT) Cooley–Tukey algorithm** and investigates how modern parallel hardware can significantly accelerate its execution.

The implementations are designed for square datasets of size:

```
N × N
```

where **N is a power of two**.

# Implementations

## 1. Serial FFT

A baseline implementation of the Radix-2 Cooley–Tukey FFT written in C.

Features:

- Iterative FFT
- Recursive FFT (for comparison)
- 2D FFT computed by:
  - FFT on rows
  - Matrix transpose
  - FFT on columns
- Synthetic dataset generation
- Timing measurements

Compile:

```bash
gcc fft_serial.c -o fft_serial.out -lm
```

Run:

```bash
./fft_serial.out 1024
```

---

## 2. OpenMP FFT

Parallel CPU implementation using OpenMP.

Parallelization is applied across independent row and column FFT computations.

Compile:

```bash
gcc fft_openmp.c -o fft_openmp.out -lm -fopenmp
```

Run:

```bash
./fft_openmp.out 1024 8
```

where

- first argument = FFT size
- second argument = number of OpenMP threads

---

## 3. CUDA FFT

GPU implementation of the FFT using custom CUDA kernels.

Main stages include:

- bit-reversal permutation
- butterfly computation
- row transforms
- matrix transpose
- column transforms

Compile:

```bash
nvcc -O2 -arch=sm_75 -o fft_cuda.out fft_cuda.cu
```

Run:

```bash
./fft_cuda.out 1024
```

---

## 4. CUDA FFT (Shared Memory)

Optimized CUDA implementation that utilizes shared memory to reduce global memory accesses during butterfly operations.

Compile:

```bash
nvcc -O2 -arch=sm_75 -o fft_cuda_shared.out fft_cuda_shared.cu
```

---

## 5. cuFFT

Reference implementation using NVIDIA's highly optimized cuFFT library.

This serves as a performance baseline against the custom CUDA implementations.

Compile:

```bash
nvcc fft_cufft.cu -lcufft -o fft_cufft.out
```

# Algorithm

The implementation follows the classic **Radix-2 Decimation-in-Time Cooley–Tukey FFT**.

For a 2D transform:

1. Generate the input matrix.
2. Perform 1D FFT on every row.
3. Transpose the matrix.
4. Perform 1D FFT on every column.
5. Compute the frequency spectrum.

The CUDA versions execute these operations using GPU kernels, while the OpenMP version distributes independent transforms across CPU threads.

---

# Benchmarking

The repository includes Python scripts for benchmarking every implementation.

Available benchmarks include:

- Serial FFT
- OpenMP FFT
- CUDA FFT
- CUDA Shared Memory FFT
- cuFFT


Results are stored as CSV files in the `benchmark/` directory for further analysis and visualization.

---

# Requirements

## CPU

- GCC
- OpenMP

## GPU

- NVIDIA GPU
- CUDA Toolkit
- NVCC compiler

## Python

Used only for benchmarking.
