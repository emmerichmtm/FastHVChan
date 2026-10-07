# Numerical hypervolume benchmarks: 4D–6D

The main comparison uses three solvers: **Numerical Chan (Numba)**, the numerical implementation with the local compression schedule proved in the report; **Chan d/3 (Python)**; and **Chan d/2 (Python)**. In the raw records their keys are `local_compiled`, `chan_dby3` and `chan_dby2`.

There are **25 inputs** and 75 selected solver/input rows: 56 complete, 18 timed out, and 1 not run. **n is the number of input points**, each defining an origin-anchored box.

Spherical fronts use n = 64, 128, 256, 512, 1024 in 4D–6D, plus n = 4096 in 4D. Simplex fronts use n = 64, 256, 1024 in each dimension. All points are nondominated. Smaller inputs are prefixes of the same seeded sample within each dimension/family.

**Table entries are median seconds over three warm calls.** **TO** means the worker exceeded its 45-second budget for startup, imports, one first call and all three warm calls together; it does not mean that one call took 45 seconds. **NR** means not run after two consecutive timeouts at smaller sizes. Neither TO nor NR supplies a warm median.

## Sphere fronts

| d | n | Numerical Chan (Numba) | Chan d/3 (Python) | Chan d/2 (Python) |
|---:|---:|---:|---:|---:|
| 4 | 64 | 0.0518 | 0.1243 | 0.0245 |
| 4 | 128 | 0.1431 | 0.3472 | 0.0684 |
| 4 | 256 | 0.3427 | 0.8913 | 0.1622 |
| 4 | 512 | 0.8485 | 2.1206 | 0.3859 |
| 4 | 1024 | 3.4890 | 4.7949 | 0.8810 |
| 4 | 4096 | TO | TO | 4.5595 |
| 5 | 64 | 0.2208 | 0.6844 | 0.0878 |
| 5 | 128 | 0.6220 | 1.9354 | 0.2568 |
| 5 | 256 | 1.9115 | 6.0686 | 0.8202 |
| 5 | 512 | 5.2159 | TO | 2.0171 |
| 5 | 1024 | TO | TO | 5.0315 |
| 6 | 64 | 0.7754 | 3.0187 | 0.2772 |
| 6 | 128 | 3.0644 | 10.9978 | 1.2158 |
| 6 | 256 | 10.3680 | TO | 3.9232 |
| 6 | 512 | TO | TO | TO |
| 6 | 1024 | TO | NR | TO |

## Simplex fronts

| d | n | Numerical Chan (Numba) | Chan d/3 (Python) | Chan d/2 (Python) |
|---:|---:|---:|---:|---:|
| 4 | 64 | 0.0589 | 0.1566 | 0.0262 |
| 4 | 256 | 0.3693 | 0.9423 | 0.1621 |
| 4 | 1024 | 4.0905 | 5.4955 | 0.9622 |
| 5 | 64 | 0.2168 | 0.7958 | 0.1137 |
| 5 | 256 | 2.1954 | 7.7507 | 0.9182 |
| 5 | 1024 | TO | TO | 6.1703 |
| 6 | 64 | 0.8562 | 3.2904 | 0.3552 |
| 6 | 256 | TO | TO | 5.2442 |
| 6 | 1024 | TO | TO | TO |

## What the comparison shows

Numerical Chan (Numba) was faster than Chan d/3 (Python) on 16/16 complete pairs; the median reference/numerical time ratio was 2.88×. Chan d/2 (Python) was faster on all 18 complete pairs. These comparisons exclude incomplete workers and do not establish a universal speedup.

At **4D, n = 4096**, Numerical Chan (Numba) returned its first value in 42.565 s, versus 23.367 s for Chan d/3 (Python). Neither finished the warm protocol. These single first-call observations can include compiled-kernel initialization and cache loading; they are not warm medians, but show why the smaller-input results should not be extrapolated to all sizes.

**Compilation matters:** this compares compiled numerical kernels with Python implementations of the references. It does not isolate the effect of the mathematical formulation or show an improved complexity exponent.

## Measurement and validation

Each call builds a fresh solver and includes input conversion and preprocessing. Warm medians exclude imports and first-call costs; existing compilation caches are retained. Workers run sequentially with no dominance prefilter or fast-math, and all compression defaults remain enabled. Only one seeded front per dimension/family and three warm repetitions were measured, under uncontrolled ordinary machine load.

Environment: Windows 11, AMD Ryzen 7 5800H (16 logical CPUs), Python 3.12.14, NumPy 2.5.3 and Numba 0.68.0.

All **242 returned values** from the three selected solvers, including available partial results, agree with the available Chan d/2 reference: **0 mismatches**, maximum scaled error 3.33e-16. These are cross-implementation checks; exact small-instance tests are separate.

The [archived full comparison](RESULTS_ALL_VARIANTS.md) preserves the earlier compiled variant, compression counters and partial-call tables. All original [inputs and raw records](results/) are unchanged. See [README.md](README.md) for reproduction and audit commands.
