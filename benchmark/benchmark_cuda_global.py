import subprocess
import re
import csv
import os

EXECUTABLE = "./fft_cuda_global.out"
##N_VALUES = [64, 128, 256, 512, 1024, 2048, 4096]
N_THREADS_PER_BLOCK = [16, 32, 64, 128, 256, 512]
RUNS = 100
OUTPUT_CSV = "results_cuda_global.csv"


with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["NThreadsPerBlock", "Run", "ExecutionTime"])

    for N_tpb in N_THREADS_PER_BLOCK:
        for run in range(RUNS):
            proc = subprocess.run(
                [EXECUTABLE, str(N_tpb)],
                capture_output=True,
                text=True,
                env={**os.environ, "OMP_NUM_THREADS": str(N_tpb)},
            )
            time = proc.stdout.strip()
            writer.writerow([N_tpb, run, time])
        print(f"Done, threads per block={N_tpb}")

print("Done!")