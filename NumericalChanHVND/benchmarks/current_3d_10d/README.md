# Current 3D-10D benchmark

This experiment uses the existing implementations without changing algorithms:

- 3D: numerical Python sweep versus the compiled Numba sweep.
- 4D: numerical Python, the current whole-contraction JIT backend, and the
  earlier hybrid backend as an additional comparison.
- 5D-10D: numerical Python versus the existing hybrid Numba backend.
- Every dimension: the repository's unmodified Python Chan d/3 and d/2 solvers.

The main comparison has two sizes per dimension and two input families,
plus one direct-magnitude sphere case per dimension: 40 datasets altogether.
Positive sphere corners are mutually nondominated; tied grids deliberately
retain duplicate and dominated generators. No dominance prefilter is enabled.
Higher-dimensional runs use smaller inputs to keep the experiment bounded.
They do not measure scalability to large inputs or establish asymptotic bounds.

From the repository root, with NumPy and Numba installed:

```bash
python NumericalChanHVND/benchmarks/current_3d_10d/benchmark_dimensions.py --out NumericalChanHVND/benchmarks/current_3d_10d/reproduction
python NumericalChanHVND/benchmarks/current_3d_10d/verify_recorded_values.py --results NumericalChanHVND/benchmarks/current_3d_10d/reproduction
python NumericalChanHVND/benchmarks/current_3d_10d/summarize_results.py --help
```

The runner refuses to overwrite existing timings. Run it with `--help` for
size-independent repeat, worker-budget and case-filter options. The saved
`results` directory contains the measured datasets and outputs. See
[RESULTS.md](RESULTS.md) for the complete comparison.
Keep custom output directories inside the repository, as in the example:
the measured runner records source and dataset hashes relative to that root.

Workers execute sequentially in fresh processes. Each constructs fresh solver
instances, records one first call, then measures three complete warm calls.
Calls include input preprocessing and necessary API coordinate transformations.
The total 45-second worker budget includes imports, first use and repetitions.
Partial samples from a timeout are retained but are not promoted to a median.
No ratio is calculated from a timeout. Existing disk caches are retained:
first-use measurements can include cache loading or compilation and are not
empty-cache compiler timings. No fastmath, algorithm parallelism or CPU
affinity is enabled.

All available values, including first calls and partial timeout samples, are
checked against the Python Chan d/2 result at relative tolerance `2e-9` and
absolute tolerance `2e-10`. The separate post-timing oracle checks cases with
at most 12 points using exact rational inclusion-exclusion. It represents each
saved binary64 coordinate exactly as a Fraction, independently of all four
algorithms. Run this oracle after benchmarking to avoid CPU contention.

Source hashes, interpreter/library versions, first-use times, every repetition,
algorithm counters, Numba signatures, and explicit worker statuses are archived.
The median ratios describe only completed matched cases, which preferentially
exclude the slowest cases. HV and magnitude timings are reported separately.
