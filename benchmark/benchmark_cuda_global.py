import subprocess
import re
import csv
import os

EXECUTABLE = "../fft_cuda_global.out"
N_VALUES = [512, 1024, 2048, 4096]
#N_THREADS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
RUNS = 100
OUTPUT_CSV = "results_cuda_global.csv"


with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "Run", "TotalTime", "ExecutionTime", "TimeHostToDevice", "TimeDeviceToHost"])

    for N in N_VALUES:
        # for nth in N_THREADS:
            for run in range(RUNS):
                proc = subprocess.run(
                    [EXECUTABLE, str(N)],
                    capture_output=True,
                    text=True,
                    env={**os.environ, "N": str(N)},
                )
                parts = proc.stdout.strip().split()

                total_time, exec_time, time_htd, time_dth = parts[0], parts[1], parts[2], parts[3]

                writer.writerow([N, run, total_time, exec_time, time_htd, time_dth])

            print(f"Done N={N}")

print("Done!")
