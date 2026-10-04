# Compression with unchanged defaults

This structural probe demonstrates that the compiled implementation executes
compression under its default policy. No compression threshold or interval is
forced. The completed run contains **six compressions**, and all three solvers
return the same hypervolume, **6,372,330,876**.

The [raw result](results/natural_compression_probe.json) retains all 1,552 input
points, events, counters and observed times. The
[portable script](natural_compression_probe.py) reproduces the construction.

## Input construction

For each of the six coordinate pairs in four dimensions, create 256 anchored
boxes with active extents `(t, 257-t)`, for `t = 1,...,256`, and extent 300 in the
other two coordinates. These 1,536 boxes become six pair constraints at the
root. They leave long cutoff arrays despite the small number of remaining
hard boxes.

Add 16 four-active corners. With `i = 0,...,15` and `j = 3i mod 16`, their
extents are `(161+i, 176-i, 161+j, 176-j)`. Their equal coordinate sums make
them mutually nondominated. Every coordinate exceeds 128.5, so no staircase
box covers one of these corners. This is a constructed family with shared
extents, not a random-input performance sample or a general-position claim.

## Recorded outcome

| Solver | Hypervolume | Compressions | Observed computation time |
|---|---:|---:|---:|
| Compiled default | 6,372,330,876 | 6 | 2.432 s |
| Python Chan d/3 default | 6,372,330,876 | 0 | 1.235 s |
| Python Chan d/2 | 6,372,330,876 | — | 0.459 s |

These are single first-use computations in fresh sequential workers, with
existing compilation caches. They are **not warm medians**. Each worker had a
45-second budget including imports, cache loading and computation. All
completed; all recorded values agree exactly as floating-point numbers.

The compiled run used its default interval of four levels and factor 8. It
visited 47 nodes, emitted 6,358 raw terms, performed 1,352 variable
eliminations, and reached a peak of 296 retained terms. Every one of the six
recorded compression starts exceeded its size threshold. For example, the
first had six hard boxes and state size 1,710 against a threshold of 1,024.
The state-size measure counts stored array entries plus terms.

Compression does not necessarily shrink that representation immediately:
all six compressions increased the stored size in this run. The first
compression produced 296 terms with total size 6,936. Rank coarsening therefore
does not guarantee fewer stored terms or entries. This run
therefore establishes that the default compression path is live and returns
the correct value; it does not establish that compression improves time or
space. The numerical and step-function representations use different size
measures, explaining why identical policy constants need not produce the
same decisions in the two Chan d/3 implementations.

The final `max_generation` is **1**: the six compressions occur in different
branches, with at most one on any path. The event field named `generation`
stores the run's global high-water counter before each compression, not the
current branch's generation. Future script output names this field
`max_generation_seen`. Repeated compression on one path is covered by
the separate correctness suite, not demonstrated by this probe.

## Reproduction and provenance

From the repository root, using the NumPy/Numba environment:

```sh
python NumericalChanHVND/benchmarks/compiled_2d_10d/natural_compression_probe.py \
  --steps 256 --hard 16 --timeout 45 --out natural_compression_rerun.json
```

The script refuses to overwrite an existing result. Its default output is
`results/natural_compression_probe.json` beside the script. The published
result was recorded on 2026-10-04 at 16:38 UTC using the original local draft;
the portable version changes paths and adds source/environment metadata and
the overwrite guard, and clarifies the event-field name described above.
It preserves the input construction and solver calls.

The original result is copied byte for byte. A post-run source audit found
that all five solver files still matched the hashes captured by the
[main benchmark](results/timings.json); this is a post-run check, not metadata
originally captured by this probe. The two new solver hashes are:

```text
numerical_chan_compiled.py
b9679a4dbc264b71132ad8555ecb81d20855c476c3362ec020d0809f4bc3e942
compiled_terms.py
f14154b7c1c452d0990d5643fdc87b19b5c962d4ce3f7fb7dffa1551e481d2b4
```

Raw-result SHA-256:
`a2929a07a91e9d651464b40223c5700c1f48f1831bec16d6c98d5589e7487520`.
Future runs of the portable script record their own source hashes and
environment directly in the JSON.
