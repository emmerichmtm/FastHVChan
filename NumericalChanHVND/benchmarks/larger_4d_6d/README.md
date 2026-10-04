# Larger 4D–6D benchmarks

This comparison includes the **local-checkpoint compiled solver covered by
the research report's complexity proof**, the earlier global-interval
compiled solver, and Python Chan d/3 and d/2.

Spherical fronts use 64, 128, 256, 512, and 1,024 points in every dimension
4–6, plus 4,096 points in 4D. A second simplex family uses 64, 256, and
1,024 points in each dimension. Inputs are nested prefixes within each
dimension/family. All generators lie on a positive nondominated front.

From the repository root, using a NumPy/Numba environment:

```sh
python NumericalChanHVND/benchmarks/larger_4d_6d/benchmark_larger.py --out fresh-results
python NumericalChanHVND/benchmarks/larger_4d_6d/summarize_larger.py --results fresh-results
```

The default worker budget is 45 seconds for startup, imports, the first
call, and three warm calls together. The script saves every returned value,
timing, and counter. Partial workers are not reported as a complete warm
median. After two consecutive worker timeouts in a solver/dimension/family
series, larger sizes in that series are marked **not run**, not timed out.
Neither status supplies an inferred single-call runtime.

Workers run sequentially. Every call builds a fresh solver and includes
input conversion and preprocessing. Compilation caches are retained; the
first call is recorded separately. All compression defaults remain enabled.
There is no dominance prefilter, fast-math, or parallel numerical execution.

The input dataset and source hashes identify precisely what was measured.
Every available value is compared with an available Chan d/2 value,
including returned values from partial workers. Large inputs are not
validated using exponential inclusion–exclusion; the separate small-input
exact verification remains applicable to the implementation.

See [RESULTS.md](RESULTS.md) and the saved [inputs and records](results/).
The retained published run can be audited without rerunning the solvers:

```sh
python NumericalChanHVND/benchmarks/larger_4d_6d/audit_results.py
```

The audit checks source hashes, nested inputs, complete-worker medians,
timeout escalation, and every returned value against the available reference.
