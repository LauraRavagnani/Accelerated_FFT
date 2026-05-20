#!/bin/bash

# Define the values of NTHREADS to test
THREAD_VALUES=(1 2 4 8 16)

# N is fixed — pass as argument or default to 512
N="${1:-512}"

# Create or clear the output file
echo "NThreads,Run,ExecutionTime" > results_omp_4096.csv

for NTHREADS in "${THREAD_VALUES[@]}"
do
    for RUN in $(seq 1 10)
    do
        # Run the program and capture only the time value
        TIME=$(OMP_NUM_THREADS=$NTHREADS ./fft_openmp.out $N | grep -oP '\d+\.\d+')

        # Save to CSV
        echo "$NTHREADS,$RUN,$TIME" >> results_omp_4096.csv
    done
    echo "Done NTHREADS=$NTHREADS"
done

echo "Done!"