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

.PHONY: all test check bench clean

all: libfasthvchan.a $(TEST)

libfasthvchan.a: fasthvchan.o
	$(AR) rcs $@ $^

fasthvchan.o: fasthvchan.c fasthvchan.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST): test_fasthvchan.c fasthvchan.c fasthvchan.h
	$(CC) $(CFLAGS) test_fasthvchan.c fasthvchan.c -o $@ $(LDLIBS)

# Self-contained: checks the recursion against brute-force inclusion-exclusion.
test: $(TEST)
	./$(TEST)

# Differential: replays cases computed by the Python implementation.
check: $(TEST)
	FASTHVCHAN_PATH=$(FASTHVCHAN_PATH) $(PYTHON) gen_cases.py cases.txt 300
	./$(TEST) --check cases.txt

bench: $(TEST)
	./$(TEST) --bench

clean:
	$(RM) fasthvchan.o libfasthvchan.a $(TEST) cases.txt
