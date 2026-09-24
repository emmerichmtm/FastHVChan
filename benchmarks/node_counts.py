"""Recursion-tree size of the Section-2 (d/2) and Section-4.2 (d/3) solvers.

Node counts measure the size of the recursion tree itself, so they are
independent of language and of per-node constant factors -- they are the
quantity the two complexity bounds actually bound.  This script runs both
solvers over the same dataset families and seed scheme as ``crossbench.py``,
on a denser size ladder (see ``PLAN``) so that the fitted growth exponents mean
something.

    python benchmarks/node_counts.py [--quick] [--out FILE]

Writes JSON that benchmarks/make_bench_tex.py turns into the report's
"Recursion size" table.
"""

import argparse
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from chan_hypervolume import Box, ChanMeasure
from chan_orthant_dby3 import ChanOrthantMeasure
from crossbench import DATASETS, REF

# Node counting needs no repetitions and no timing, so it can afford a denser
# size ladder than crossbench.py uses.  That matters: a growth exponent fitted
# through the two sizes per dimension of the timing ladder is a two-point
# slope, far too noisy to separate d/2 from d/3.  Each ladder below spans at
# least a 4x range in n.
PLAN = {3: [50, 100, 200, 400, 800],
        4: [30, 60, 120, 240],
        5: [20, 40, 80, 160],
        7: [12, 25, 50, 100],
        9: [10, 20, 40],
        10: [10, 20, 40]}


def fit_exponent(ns, values):
    """Least-squares slope in log-log scale, or None if degenerate."""
    pairs = [(n, v) for n, v in zip(ns, values) if n > 0 and v > 0]
    if len(pairs) < 2:
        return None
    lx = [math.log(n) for n, _ in pairs]
    ly = [math.log(v) for _, v in pairs]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    denom = sum((a - mx) ** 2 for a in lx)
    if denom == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="drop the largest size per dimension, and d <= 7")
    parser.add_argument("--seed", type=int, default=20260821,
                        help="same generator and seed scheme as crossbench.py")
    parser.add_argument("--out", default=os.path.join(HERE, "results",
                                                      "nodecounts.json"))
    args = parser.parse_args()

    dims = [d for d in sorted(PLAN) if not args.quick or d <= 7]
    sizes = {d: (PLAN[d][:-1] if args.quick else PLAN[d]) for d in dims}

    records = []
    print("%-10s %3s %5s %11s %11s %7s" %
          ("dataset", "d", "n", "nodes d/2", "nodes d/3", "ratio"))
    for dataset_name, generator in DATASETS:
        for dim in dims:
            for n in sizes[dim]:
                rng = random.Random(args.seed + 1000 * dim + n)
                points = generator(n, dim, rng)
                ref = tuple(REF for _ in range(dim))
                lo = tuple(min(p[k] for p in points) for k in range(dim))

                solver2 = ChanMeasure(dim)
                solver2.complement_measure([Box(p, ref) for p in points],
                                           Box(lo, ref))
                solver3 = ChanOrthantMeasure(dim)
                solver3.complement_measure(points, lo, ref)

                ratio = (solver2.node_count / solver3.node_count
                         if solver3.node_count else None)
                records.append({
                    "dataset": dataset_name, "dim": dim, "n": n,
                    "nodes_dby2": solver2.node_count,
                    "nodes_dby3": solver3.node_count,
                    "terms_high_water": solver3.term_high_water,
                    "compressions": solver3.compress_count,
                })
                print("%-10s %3d %5d %11d %11d %6s" %
                      (dataset_name, dim, n, solver2.node_count,
                       solver3.node_count,
                       "%.1fx" % ratio if ratio else "--"))
                sys.stdout.flush()

    # Fitted exponents per (dataset, dim), which is what the bounds predict.
    exponents = []
    for dataset_name, _ in DATASETS:
        for dim in dims:
            rows = [r for r in records
                    if r["dataset"] == dataset_name and r["dim"] == dim]
            if len(rows) < 2:
                continue
            ns = [r["n"] for r in rows]
            exponents.append({
                "dataset": dataset_name, "dim": dim,
                "exp_dby2": fit_exponent(ns, [r["nodes_dby2"] for r in rows]),
                "exp_dby3": fit_exponent(ns, [r["nodes_dby3"] for r in rows]),
            })

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(args.out, "w") as handle:
        json.dump({"seed": args.seed, "dims": dims,
                   "sizes": {str(k): v for k, v in sizes.items()},
                   "records": records, "exponents": exponents},
                  handle, indent=1, sort_keys=True)
    print("\nwrote %s (%d records)" % (args.out, len(records)))


if __name__ == "__main__":
    main()
