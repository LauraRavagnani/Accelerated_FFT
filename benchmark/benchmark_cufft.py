import subprocess
import re
import csv
import os

EXECUTABLE = "../fft_cufft.out"
N_VALUES = [512, 1024, 2048, 4096]
RUNS = 100
OUTPUT_CSV = "results_cufft.csv"

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "Run", "ExecutionTime"])

    for N in N_VALUES:
        # for nth in N_THREADS:
            for run in range(RUNS):
                proc = subprocess.run(
                    [EXECUTABLE, str(N)],
                    capture_output=True,
                    text=True,
                    env={**os.environ, "N": str(N)},
                )
                time = proc.stdout.strip()
                writer.writerow([N, run, time])
            print(f"Done N={N}")

print("Done!")