import subprocess
import re
import csv
import os

EXECUTABLE = "../fft_cuda.out"
N_VALUES = [512, 1024, 2048, 4096]
RUNS = 100
OUTPUT_CSV = "results_cuda.csv"

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "Run", "TotalTime", "ExecutionTime", "TimeHostToDevice", "TimeDeviceToHost", "GFlops", "GbpsHtD", "GbpsDtH"])

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

                total_time, exec_time, time_htd, time_dth, gflops, gbps_htd, gbps_dth = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]

                writer.writerow([N, run, total_time, exec_time, time_htd, time_dth, gflops, gbps_htd, gbps_dth])

            print(f"Done N={N}")

print("Done!")
