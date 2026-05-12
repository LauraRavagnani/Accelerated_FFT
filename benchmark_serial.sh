#!/bin/bash

#gcc -O3 fft_serial.c -o fft_bench.out -lm

# Define the values of N to test
N_VALUES=(64 128 256 512 1024 2048 4096)

# Create or clear the output file
echo "N,ExecutionTime" > results_ditfft2.csv

# Create or clear the output file
# echo "N,ExecutionTime" > results_iterative_fft.csv

for N in "${N_VALUES[@]}"
do
    #echo "Running N=$N..."
    # Run the program and capture only the time value
    TIME=$(./fft_serial.out $N | grep -oP '\d+\.\d+')
    
    # Save to CSV
    echo "$N,$TIME" >> results_ditfft2.csv

    # Save to CSV
    # echo "$N,$TIME" >> results_iterative_fft.csv
done

echo "Done!"