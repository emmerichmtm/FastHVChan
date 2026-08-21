# Plain '=' not '?=': make has a built-in CC=cc that ?= will not override.
# A command-line override (make CC=clang) still wins.
CC       = gcc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -pedantic
LDLIBS  ?= -lm
PYTHON  ?= python

# Where the Python original lives, for the differential oracle.
FASTHVCHAN_PATH ?= ..

EXE :=
ifeq ($(OS),Windows_NT)
EXE := .exe
endif

TEST := test_fasthvchan$(EXE)
TESTBL := test_hv_baselines$(EXE)
DRIVER := bench_driver$(EXE)

.PHONY: all test check bench clean

all: libfasthvchan.a $(TEST) $(TESTBL) $(DRIVER)

libfasthvchan.a: fasthvchan.o hv_baselines.o
	$(AR) rcs $@ $^

fasthvchan.o: fasthvchan.c fasthvchan.h
	$(CC) $(CFLAGS) -c $< -o $@

hv_baselines.o: hv_baselines.c hv_baselines.h fasthvchan.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST): test_fasthvchan.c fasthvchan.c fasthvchan.h
	$(CC) $(CFLAGS) test_fasthvchan.c fasthvchan.c -o $@ $(LDLIBS)

$(TESTBL): test_hv_baselines.c hv_baselines.c hv_baselines.h fasthvchan.c fasthvchan.h
	$(CC) $(CFLAGS) test_hv_baselines.c hv_baselines.c fasthvchan.c -o $@ $(LDLIBS)

# Self-contained: checks every algorithm against brute-force inclusion-exclusion.
test: $(TEST) $(TESTBL)
	./$(TEST)
	./$(TESTBL)

# Differential: replays cases computed by the Python implementation.
check: $(TEST)
	FASTHVCHAN_PATH=$(FASTHVCHAN_PATH) $(PYTHON) gen_cases.py cases.txt 300
	./$(TEST) --check cases.txt

bench: $(TEST)
	./$(TEST) --bench

# Used by benchmarks/crossbench.py; times one dataset file.
$(DRIVER): bench_driver.c fasthvchan.c fasthvchan.h hv_baselines.c hv_baselines.h
	$(CC) $(CFLAGS) bench_driver.c fasthvchan.c hv_baselines.c -o $@ $(LDLIBS)

clean:
	$(RM) fasthvchan.o hv_baselines.o libfasthvchan.a $(TEST) $(TESTBL) $(DRIVER) cases.txt
