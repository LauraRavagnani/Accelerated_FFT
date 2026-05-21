import subprocess
import re
import csv

EXECUTABLE = "./fft_serial.out"
N_VALUES = [64, 128, 256, 512, 1024, 2048, 4096]
RUNS = 10
iterative_mode = False

if iterative_mode:
    OUTPUT_CSV = "results_iterative_fft.csv"
else:
    OUTPUT_CSV = "results_ditfft2.csv"


with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "Run", "ExecutionTime"])

    for N in N_VALUES:
        for run in range(RUNS):
            proc = subprocess.run(
                [EXECUTABLE, str(N)],
                capture_output=True,
                text=True,
            )
            time = proc.stdout.strip()
            writer.writerow([N, run, time])

        print(f"Done N={N}")

print("Done!")