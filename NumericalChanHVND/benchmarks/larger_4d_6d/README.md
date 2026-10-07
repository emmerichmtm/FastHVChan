# Larger 4D–6D benchmarks

[RESULTS.md](RESULTS.md) is the main comparison. Its tables contain only:

- **Numerical Chan (Numba)**: the local compression schedule covered by the report's proof.
- **Chan d/3 (Python)**: the existing Section 4.2 reference.
- **Chan d/2 (Python)**: the existing Section 2 reference.

The compiled numerical solver is compared with interpreted references, so
these timings do not isolate the effect of the numerical formulation.

Spherical fronts use 64, 128, 256, 512, and 1,024 points in every dimension
4–6, plus 4,096 points in 4D. Simplex fronts use 64, 256, and 1,024 points
in each dimension. Inputs are nested prefixes within each dimension/family.
All generators lie on a positive nondominated front.

## Read or regenerate the summary

The stored measurements can be summarized without rerunning any solver:

```sh
python NumericalChanHVND/benchmarks/larger_4d_6d/summarize_larger.py
```

This regenerates only the concise Markdown report. It does not change the
paper or the raw records. The [archived full comparison](RESULTS_ALL_VARIANTS.md)
retains the earlier compiled policy, all compression counters and partial
worker timings. Regenerate that additional detail explicitly with:

```sh
python NumericalChanHVND/benchmarks/larger_4d_6d/summarize_larger.py --all-variants
```

## Reproduce or audit

From the repository root, using a NumPy/Numba environment:

```sh
python NumericalChanHVND/benchmarks/larger_4d_6d/benchmark_larger.py --out fresh-results
python NumericalChanHVND/benchmarks/larger_4d_6d/summarize_larger.py --results fresh-results
```

The runner still measures all four archived variants to preserve the original
protocol; the main summary selects the three solvers above. Each worker has
45 seconds for startup, imports, one first call and three warm calls together.
**TO** means this combined budget expired, not that a single call took 45
seconds. **NR** means not run after two consecutive timeouts in a
solver/dimension/family series. No warm median is inferred from either status.
Every returned timing and value, including partial results, remains in the
[raw records](results/).

Workers run sequentially. Every call builds a fresh solver and includes
input conversion and preprocessing. Compilation caches are retained; first
calls are recorded separately. All compression defaults remain enabled.
There is no dominance prefilter, fast-math, or parallel numerical execution.
Every available value is compared with an available Chan d/2 value. Large
inputs are not checked by exponential inclusion–exclusion; the separate
small-input exact verification remains applicable.

Audit the retained published measurements without rerunning the solvers:

```sh
python NumericalChanHVND/benchmarks/larger_4d_6d/audit_results.py
```

The audit covers all original variants, source hashes, nested inputs,
complete-worker medians, timeout escalation, and every returned value.
The original hashes describe the measured Windows checkout. Git can change
line endings, so `results/source_provenance.json` also records hashes after
normalizing CRLF to LF. The audit accepts this line-ending difference while
still checking source contents.
