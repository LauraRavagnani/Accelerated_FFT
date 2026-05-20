# import subprocess
# import re
# import statistics
# import json
# from pathlib import Path
# import csv

# # # Configuration
# EXECUTABLE = "./fft_serial.out"          # path to your compiled binary
# N_VALUES = [64, 128, 256, 512, 1024, 2048, 4096]
# RUNS_PER_N = 10                        # number of repetitions for statistical stability

# def parse_elapsed(output: str) -> float:
#     """Extract the execution time from the program's stdout."""
#     match = re.search(r"Execution time \(fft\):\s+([\d.]+)\s+seconds", output)
#     if not match:
#         raise ValueError(f"Could not parse timing from output:\n{output}")
#     return float(match.group(1))

# def benchmark(executable: str, n_values: list[int], runs: int) -> dict:
#     results = {}

#     for N in n_values:
#         times = []
#         print(f"Benchmarking N={N} ({runs} runs)...", flush=True)

#         for run in range(runs):
#             proc = subprocess.run(
#                 [executable, str(N)],
#                 capture_output=True,
#                 text=True,        # bail out if a single run takes > 2 min
#             )
            

#             t = parse_elapsed(proc.stdout)
#             times.append(t)
#             print(f"  run {run+1}: {t:.6f}s")

#         if times:
#             results[N] = {
#                 "runs":   len(times),
#                 "mean":   statistics.mean(times),
#                 "median": statistics.median(times),
#                 "stdev":  statistics.stdev(times) if len(times) > 1 else 0.0,
#                 "min":    min(times),
#                 "max":    max(times),
#                 "raw":    times,
#             }
#         else:
#             results[N] = None   # all runs failed

#     return results



# if __name__ == "__main__":
#     results = benchmark(EXECUTABLE, N_VALUES, RUNS_PER_N)
#     save_json(results)




import subprocess
import re
import csv

EXECUTABLE = "./fft_serial.out"
N_VALUES = [64, 128, 256, 512, 1024, 2048, 4096]
RUNS = 10
OUTPUT_CSV = "results_iterative_fft.csv"


with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N", "Run", "ExecutionTime"])

    for N in N_VALUES:
        for run in range(1, RUNS + 1):
            proc = subprocess.run(
                [EXECUTABLE, str(N)],
                capture_output=True,
                text=True,
            )
            time = proc.stdout.strip()
            writer.writerow([N, run, time])

        print(f"Done N={N}")

print("Done!")