import subprocess
import re
import csv
import os

EXECUTABLE = "../fft_cuda_naif.out"
##N_VALUES = [64, 128, 256, 512, 1024, 2048, 4096]
N_THREADS_PER_BLOCK = [16, 32, 64, 128, 256, 512, 1024]
RUNS = 10
OUTPUT_CSV = "results_cuda_naif.csv"


with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["NThreads", "Run", "ExecutionTime"])

    for nth in N_THREADS_PER_BLOCK:
        for run in range(RUNS):
            proc = subprocess.run(
                [EXECUTABLE, str(N), str(nth)],
                capture_output=True,
                text=True,
                env={**os.environ, "OMP_NUM_THREADS": str(nth)},
            )
            time = proc.stdout.strip()
            writer.writerow([N, nth, run, time])
        print(f"Done N={N}, threads={nth}")

print("Done!")