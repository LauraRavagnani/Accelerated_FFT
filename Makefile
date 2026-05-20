# ── Compiler & flags ──────────────────────────────────────────────────────────
CC      = gcc
CFLAGS  = -fopenmp
LDFLAGS = -lm

# ── Targets ───────────────────────────────────────────────────────────────────
TARGET  = fft_openmp.o

# # ── Default N if not passed on the command line ───────────────────────────────
# N       ?= 1024

# ── Number of threads if not passed on the command line ───────────────────────
THREADS ?= 16

# # ──────────────────────────────────────────────────────────────────────────────

# .PHONY: all run clean

all: $(TARGET)

$(TARGET): fft_openmp.c utils_openmp.c
	$(CC) $(CFLAGS) -o $(TARGET) main_omp.c $(LDFLAGS)

run: $(TARGET)
	OMP_NUM_THREADS=$(THREADS) ./$(TARGET) $(N)

clean:
	rm -f $(TARGET) *.json