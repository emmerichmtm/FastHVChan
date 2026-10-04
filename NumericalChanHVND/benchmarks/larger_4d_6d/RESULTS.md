# Larger hypervolume benchmarks: 4D–6D

25 inputs, 100 planned algorithm/input rows. Statuses: {'ok': 74, 'timeout': 25, 'not_run_after_timeouts': 1}. Value mismatches: **0**.

**n is the number of input generators**, and each generator defines one origin-anchored box. These runs measure the new local-checkpoint compiled policy as well as the earlier global-interval policy.

## Coverage and interpretation

Spherical fronts use n = 64, 128, 256, 512, 1024 in 4D, 5D and 6D, plus n = 4096 in 4D. A second simplex family uses n = 64, 256, 1024 in each dimension. Within a dimension/family, smaller inputs are prefixes of the largest seeded sample. All points are on a positive nondominated front; no dominance filter is used.

Local compiled statuses: **{'ok': 18, 'timeout': 7}**. 4 completed local inputs executed compression.

| Dimension | Sphere: 3 warm calls | Sphere: any returned value | Simplex: 3 warm calls | Simplex: any returned value |
|---|---:|---:|---:|---:|
| 4 | 1024 | 4096 | 1024 | 1024 |
| 5 | 512 | 1024 | 256 | 1024 |
| 6 | 256 | 512 | 64 | 256 |

Ratios below use only pairs with all three warm calls complete; censored larger cases are excluded, so these ratios should not be treated as universal speedups.

- Against Earlier compiled: local compiled wins 8/18 complete pairs; median reference/local time ratio 0.991×.
- Against Chan d/3 Python: local compiled wins 16/16 complete pairs; median reference/local time ratio 2.88×.
- Against Chan d/2 Python: local compiled wins 0/18 complete pairs; median reference/local time ratio 0.417×.

At **4D, n=4096**, local compiled returned its first value in 42.565 s, versus 23.367 s for Python Chan d/3. Neither completed the full warm protocol. Thus the advantages on complete smaller pairs do not establish an advantage at larger n. These are first-call observations, including possible compiled-kernel initialization, not warm medians.

## Sphere fronts

Warm median **seconds**, not milliseconds.

| d | n | Local compiled | Earlier compiled | Chan d/3 Python | Chan d/2 Python |
|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 0.0518 | 0.0541 | 0.1243 | 0.0245 |
| 4 | 128 | 0.1431 | 0.1417 | 0.3472 | 0.0684 |
| 4 | 256 | 0.3427 | 0.3366 | 0.8913 | 0.1622 |
| 4 | 512 | 0.8485 | 0.9259 | 2.1206 | 0.3859 |
| 4 | 1024 | 3.4890 | 4.9178 | 4.7949 | 0.8810 |
| 4 | 4096 | timeout (0/3 warm) | timeout (0/3 warm) | timeout (0/3 warm) | 4.5595 |
| 5 | 64 | 0.2208 | 0.2145 | 0.6844 | 0.0878 |
| 5 | 128 | 0.6220 | 0.6149 | 1.9354 | 0.2568 |
| 5 | 256 | 1.9115 | 1.8885 | 6.0686 | 0.8202 |
| 5 | 512 | 5.2159 | 5.2239 | timeout (1/3 warm) | 2.0171 |
| 5 | 1024 | timeout (1/3 warm) | timeout (0/3 warm) | timeout (0/3 warm) | 5.0315 |
| 6 | 64 | 0.7754 | 0.7949 | 3.0187 | 0.2772 |
| 6 | 128 | 3.0644 | 3.0222 | 10.9978 | 1.2158 |
| 6 | 256 | 10.3680 | 10.2754 | timeout (0/3 warm) | 3.9232 |
| 6 | 512 | timeout (0/3 warm) | timeout (0/3 warm) | timeout (0/3 warm) | timeout (2/3 warm) |
| 6 | 1024 | timeout (0/3 warm) | timeout (0/3 warm) | not run after timeouts | timeout (0/3 warm) |

## Simplex fronts

Warm median **seconds**, not milliseconds.

| d | n | Local compiled | Earlier compiled | Chan d/3 Python | Chan d/2 Python |
|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 0.0589 | 0.0543 | 0.1566 | 0.0262 |
| 4 | 256 | 0.3693 | 0.3574 | 0.9423 | 0.1621 |
| 4 | 1024 | 4.0905 | 5.4003 | 5.4955 | 0.9622 |
| 5 | 64 | 0.2168 | 0.2385 | 0.7958 | 0.1137 |
| 5 | 256 | 2.1954 | 2.2019 | 7.7507 | 0.9182 |
| 5 | 1024 | timeout (0/3 warm) | timeout (0/3 warm) | timeout (0/3 warm) | 6.1703 |
| 6 | 64 | 0.8562 | 0.8410 | 3.2904 | 0.3552 |
| 6 | 256 | timeout (2/3 warm) | timeout (2/3 warm) | timeout (0/3 warm) | 5.2442 |
| 6 | 1024 | timeout (0/3 warm) | timeout (0/3 warm) | timeout (0/3 warm) | timeout (0/3 warm) |

## Compression in the local compiled solver

Counters are from the completed first call even if later repetitions exceeded the worker budget.

| Input | Worker status | Checkpoints | Compressions | Peak terms |
|---|---|---:|---:|---:|
| sphere-d4-n64-hv | ok | 19 | 0 | 8 |
| sphere-d4-n128-hv | ok | 112 | 0 | 9 |
| sphere-d4-n256-hv | ok | 209 | 0 | 10 |
| sphere-d4-n512-hv | ok | 276 | 3 | 81 |
| sphere-d4-n1024-hv | ok | 1467 | 582 | 210 |
| sphere-d4-n4096-hv | timeout | 3160 | 2582 | 750 |
| sphere-d5-n64-hv | ok | 129 | 0 | 15 |
| sphere-d5-n128-hv | ok | 396 | 0 | 12 |
| sphere-d5-n256-hv | ok | 693 | 0 | 15 |
| sphere-d5-n512-hv | ok | 2201 | 4 | 108 |
| sphere-d5-n1024-hv | timeout | 8496 | 1087 | 609 |
| sphere-d6-n64-hv | ok | 442 | 0 | 14 |
| sphere-d6-n128-hv | ok | 1155 | 0 | 25 |
| sphere-d6-n256-hv | ok | 2538 | 0 | 23 |
| sphere-d6-n512-hv | timeout | 14578 | 0 | 31 |
| simplex-d4-n64-hv | ok | 23 | 0 | 10 |
| simplex-d4-n256-hv | ok | 205 | 0 | 10 |
| simplex-d4-n1024-hv | ok | 1609 | 649 | 196 |
| simplex-d5-n64-hv | ok | 150 | 0 | 14 |
| simplex-d5-n256-hv | ok | 696 | 0 | 16 |
| simplex-d5-n1024-hv | timeout | 9314 | 1143 | 521 |
| simplex-d6-n64-hv | ok | 432 | 0 | 18 |
| simplex-d6-n256-hv | timeout | 2893 | 0 | 25 |

## Censored workers

A worker has a 45-second budget for imports, one first call, and three warm calls. A timeout is **not** a lower bound on the runtime of one call. Partial calls that returned are retained below and in the raw JSON, but are not reported as a complete warm median. After two consecutive timeouts in a solver/dimension/family series, larger sizes in that series are marked not run. This resource rule does not prove that those larger cases would time out.

| Input | Algorithm | Status | First call (s) | Completed warm calls (s) |
|---|---|---|---:|---|
| sphere-d4-n4096-hv | local_compiled | timeout | 42.5652 | none |
| sphere-d4-n4096-hv | global_compiled | timeout | 42.9372 | none |
| sphere-d4-n4096-hv | chan_dby3 | timeout | 23.3672 | none |
| sphere-d5-n512-hv | chan_dby3 | timeout | 16.4554 | 17.6806 |
| sphere-d5-n1024-hv | global_compiled | timeout | 31.5505 | none |
| sphere-d5-n1024-hv | local_compiled | timeout | 22.5823 | 20.0630 |
| sphere-d5-n1024-hv | chan_dby3 | timeout | 43.5353 | none |
| sphere-d6-n256-hv | chan_dby3 | timeout | 39.9976 | none |
| sphere-d6-n512-hv | chan_dby2 | timeout | 12.6784 | 12.6964, 12.6187 |
| sphere-d6-n512-hv | local_compiled | timeout | 36.8258 | none |
| sphere-d6-n512-hv | chan_dby3 | timeout | no value | none |
| sphere-d6-n512-hv | global_compiled | timeout | 36.7280 | none |
| sphere-d6-n1024-hv | chan_dby3 | not_run_after_timeouts | no value | none |
| sphere-d6-n1024-hv | local_compiled | timeout | no value | none |
| sphere-d6-n1024-hv | global_compiled | timeout | no value | none |
| sphere-d6-n1024-hv | chan_dby2 | timeout | 34.4914 | none |
| simplex-d5-n1024-hv | chan_dby3 | timeout | no value | none |
| simplex-d5-n1024-hv | local_compiled | timeout | 25.1915 | none |
| simplex-d5-n1024-hv | global_compiled | timeout | 34.3725 | none |
| simplex-d6-n256-hv | global_compiled | timeout | 13.6164 | 11.3107, 11.3699 |
| simplex-d6-n256-hv | local_compiled | timeout | 12.8929 | 11.0682, 11.0424 |
| simplex-d6-n256-hv | chan_dby3 | timeout | no value | none |
| simplex-d6-n1024-hv | global_compiled | timeout | no value | none |
| simplex-d6-n1024-hv | local_compiled | timeout | no value | none |
| simplex-d6-n1024-hv | chan_dby3 | timeout | no value | none |
| simplex-d6-n1024-hv | chan_dby2 | timeout | 43.5629 | none |

## Method and correctness

Nested positive spherical and simplex fronts. Fresh sequential workers, one first call and 3 warm calls, new solver per call. Existing compilation caches retained. Timeout includes startup/imports/all calls. No prefilter, fastmath or parallel solvers. All compression defaults retained. Partial values/times retained, never called a full median.

Environment: Windows 11, AMD Ryzen 7 5800H (16 logical CPUs), Python 3.12.14, NumPy 2.5.3 and Numba 0.68.0. The raw record retains the platform processor identifier.

Each timed call includes input conversion and preprocessing. Warm medians exclude import and first-call costs. Existing caches are retained; first-call times are not empty-cache JIT compilation measurements. Only one numerical worker runs at a time. Ordinary machine activity is uncontrolled; results use one seeded front per family/dimension and three repetitions.

For completed local workers, median slowest/fastest warm-call ratio is 1.05, maximum 1.26. No confidence interval or asymptotic exponent is inferred.

321 returned values across 91 rows were checked against Chan d/2; maximum scaled error is 4.44e-16. Every available first and warm value is checked. If the reference completed its first call but later timed out, that completed value remains usable for validation. A row without an available reference is marked unverified, not passed.

These larger cases are cross-implementation checks, not exponential exact inclusion–exclusion oracles. Exact small-instance and signed-algebra tests remain in the separate verification suites.

The local schedule is the policy covered by the report’s arithmetic theorem. The earlier global-interval solver has a different schedule. Their measurements remain separate.

Raw inputs, timings, counters and source hashes: [results](results/). Run [benchmark_larger.py](benchmark_larger.py) to reproduce with a new output directory.
