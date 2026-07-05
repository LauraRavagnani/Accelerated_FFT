import subprocess
import re
import csv
import os

EXECUTABLE = "../fft_cuda_global.out"
N = 512
N_THREADS_PER_BLOCK = [32, 64, 128, 256]
RUNS = 100
OUTPUT_CSV = "./results_tune_tpb.csv"

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["NThreadsPerBlock", "Run", "ExecutionTime"])

    for N_tpb in N_THREADS_PER_BLOCK:
        for run in range(RUNS):
            proc = subprocess.run(
                [EXECUTABLE, str(N), str(N_tpb)],
                capture_output=True,
                text=True,
                env={**os.environ, "NUM_THREADS": str(N_tpb)},
            )
            time = proc.stdout.strip()
            writer.writerow([N_tpb, run, time])
        print(f"Done, threads per block={N_tpb}")

print("Done!")

