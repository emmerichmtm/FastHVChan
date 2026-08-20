"""Scaling benchmarks: node counts and wall time of both algorithms on
spherical fronts (every point mutually non-dominated -- the standard hard
case, since nothing can be pruned before the real work starts).

Usage:  python benchmarks/benchmark.py [--quick]
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chan_hypervolume import Box, ChanMeasure
from chan_orthant_dby3 import ChanOrthantMeasure

REF = 1.2  # reference-point coordinate


def sphere_front(n, d, rng):
    pts = []
    for _ in range(n):
        v = [abs(rng.gauss(0.0, 1.0)) + 1e-9 for _ in range(d)]
        s = math.sqrt(sum(x * x for x in v))
        pts.append(tuple(x / s for x in v))
    return pts


def fit_exponent(ns, ys):
    """Least-squares slope in log-log scale."""
    lx = [math.log(n) for n in ns]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum(
        (a - mx) ** 2 for a in lx
    )


def run(plan):
    rng = random.Random(99)
    for d, counts in plan.items():
        rows = []
        print(f"== d={d}  (worst-case node exponents: d/2={d/2:.2f}, d/3={d/3:.2f})")
        for n in counts:
            pts = sphere_front(n, d, rng)
            ref = tuple(REF for _ in range(d))
            lo = tuple(min(p[k] for p in pts) for k in range(d))
            domain_vol = math.prod(b - a for a, b in zip(lo, ref))

            s2 = ChanMeasure(d)
            t0 = time.perf_counter()
            v2 = domain_vol - s2.complement_measure([Box(p, ref) for p in pts], Box(lo, ref))
            t2 = time.perf_counter() - t0

            s3 = ChanOrthantMeasure(d)
            t0 = time.perf_counter()
            v3 = domain_vol - s3.complement_measure(pts, lo, ref)
            t3 = time.perf_counter() - t0

            assert abs(v2 - v3) < 1e-8 * max(1.0, v2), (d, n, v2, v3)
            rows.append((n, s2.node_count, s3.node_count, t2, t3))
            print(
                f"  n={n:>4}  nodes: {s2.node_count:>6} vs {s3.node_count:>6}"
                f"   time: {t2:7.3f}s vs {t3:7.3f}s"
                f"   (compressions: {s3.compress_count})"
            )
        ns = [r[0] for r in rows]
        print(
            f"  fitted node exponents:  d/2-alg {fit_exponent(ns, [r[1] for r in rows]):.2f}"
            f"   d/3-alg {fit_exponent(ns, [r[2] for r in rows]):.2f}"
        )
        print(
            f"  fitted time exponents:  d/2-alg {fit_exponent(ns, [r[3] for r in rows]):.2f}"
            f"   d/3-alg {fit_exponent(ns, [r[4] for r in rows]):.2f}\n"
        )


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if quick:
        plan = {3: [16, 32, 64], 4: [12, 24, 48], 5: [8, 16]}
    else:
        plan = {
            3: [16, 32, 64, 128, 256],
            4: [12, 24, 48, 96],
            5: [8, 16, 32],
        }
    run(plan)
