"""Structural probe of the Section-4.2 (d/3) solver on the wfg_hard instances.

The wfg_hard family is built so that every point is non-dominated through only
two of its objectives, the rest being tied.  Relative to the root cell, then,
every orthant has exactly two *active* constraints -- and an orthant with at
most two active constraints is exactly what Chan's Section-4.1 simplification
absorbs (as a slab or a 2-sided orthant).  The prediction is therefore that the
d/3 recursion never cuts at all: the surviving set B-bar is empty at the root
and the whole computation is one absorption followed by one integration of a
single basic function.

This script checks that prediction and, when it holds, splits the running time
into its two phases, which isolates the cost of Lemma 4.4 integration.

    python benchmarks/wfg_hard_probe.py --data PATH [--check] [--out FILE]

PATH is the ``wfg_hard/min`` directory of

    https://github.com/renaudlr/moo-nondominated-sets

(the instances are named ``wfg_hard_d<D>n<N>_1``).  With ``--check`` the
hypervolume is also verified against the WFG baseline, which is only affordable
for the smaller instances -- see ``CHECK_MAX_N``, whose ceiling falls with the
dimension because this family is exactly the one built to defeat WFG.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from chan_orthant_dby3 import (ChanOrthantMeasure, Term, _terms_complexity,
                               integrate_terms)
from hv_baselines import hypervolume_wfg
from node_counts import fit_exponent

REF = 1.1  # every wfg_hard coordinate is in [0, 1]

# The WFG oracle is exponential, and wfg_hard is precisely the family built to
# defeat it, so the affordable ceiling drops sharply with the dimension: at
# d=10 even n=60 runs for minutes, and the cost is erratic between neighbouring
# sizes because it follows the instance's block structure rather than n alone
# (d=8, n=88 is far worse than d=8, n=120).  These are deliberately
# conservative spot checks -- the structural result (B-bar empty, a single
# node) is what the probe is really after, and it needs no oracle.
CHECK_MAX_N = {4: 200, 6: 180, 8: 60, 10: 30}
CHECK_DEFAULT = 30


def load_instance(path):
    points = [tuple(float(x) for x in line.split())
              for line in open(path) if line.strip()]
    return points


def discover(data_dir):
    """Map the instance files to (dim, n, path), sorted."""
    found = []
    for path in glob.glob(os.path.join(data_dir, "wfg_hard_d*n*")):
        match = re.search(r"wfg_hard_d(\d+)n(\d+)", os.path.basename(path))
        if match:
            found.append((int(match.group(1)), int(match.group(2)), path))
    return sorted(found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="directory holding the wfg_hard instances")
    parser.add_argument("--check", action="store_true",
                        help="verify against WFG where affordable")
    parser.add_argument("--out", default=os.path.join(HERE, "results",
                                                      "wfg_hard.json"))
    args = parser.parse_args()

    instances = discover(args.data)
    if not instances:
        raise SystemExit("no wfg_hard_d*n* instances under %s\n"
                         "Clone them from "
                         "https://github.com/renaudlr/moo-nondominated-sets "
                         "and point --data at wfg_hard/min" % args.data)

    records = []
    print("%3s %5s %8s %7s %7s %9s %9s %9s %s" %
          ("d", "n", "|B-bar|", "nodes", "active", "absorb", "integrate",
           "total", "check"))
    for dim, n, path in instances:
        points = load_instance(path)
        if len(points) != n or len(points[0]) != dim:
            print("  skipping malformed %s" % path)
            continue
        ref = tuple(REF for _ in range(dim))
        lo = tuple(min(p[k] for p in points) for k in range(dim))
        domain = 1.0
        for a, b in zip(lo, ref):
            domain *= b - a

        # How many constraints of each orthant actually cut the root cell?
        probe = ChanOrthantMeasure(dim)
        boxes = [(p, (1,) * dim) for p in points]
        active = sorted({sum(1 for k in range(dim)
                             if probe._active(p, s, k, list(lo), list(ref)))
                         for p, s in boxes})

        # Phase split: absorption, then integration of the basic function.
        start = time.perf_counter()
        absorbed = probe._absorb(boxes, [Term(1.0)], list(lo), list(ref))
        t_absorb = time.perf_counter() - start
        if absorbed is None:
            surviving, complexity, t_integrate = 0, 0, 0.0
        else:
            surviving = len(absorbed[0])
            complexity = _terms_complexity(absorbed[1])
            start = time.perf_counter()
            integrate_terms(absorbed[1], absorbed[2], absorbed[3])
            t_integrate = time.perf_counter() - start

        # Full solve, for node count and end-to-end time.
        solver = ChanOrthantMeasure(dim)
        start = time.perf_counter()
        value = domain - solver.complement_measure(points, lo, ref)
        t_total = time.perf_counter() - start

        tag = "--"
        if args.check and n <= CHECK_MAX_N.get(dim, CHECK_DEFAULT):
            want = hypervolume_wfg(points, ref)
            tag = ("ok" if abs(value - want) <= 1e-8 * max(1.0, want)
                   else "MISMATCH")

        records.append({
            "dim": dim, "n": n, "surviving": surviving,
            "active_counts": active, "nodes": solver.node_count,
            "terms_high_water": solver.term_high_water,
            "complexity": complexity, "hv": value,
            "absorb_seconds": t_absorb, "integrate_seconds": t_integrate,
            "total_seconds": t_total, "check": tag,
        })
        print("%3d %5d %8d %7d %7s %9.4f %9.4f %9.4f %s" %
              (dim, n, surviving, solver.node_count,
               ",".join(str(a) for a in active),
               t_absorb, t_integrate, t_total, tag))
        sys.stdout.flush()

    summaries = []
    for dim in sorted({r["dim"] for r in records}):
        rows = sorted((r for r in records if r["dim"] == dim),
                      key=lambda r: r["n"])
        if len(rows) < 2:
            continue
        ns = [r["n"] for r in rows]
        summaries.append({
            "dim": dim,
            "exp_total": fit_exponent(ns, [r["total_seconds"] for r in rows]),
            "exp_absorb": fit_exponent(ns, [r["absorb_seconds"] for r in rows]),
            "exp_integrate": fit_exponent(
                ns, [r["integrate_seconds"] for r in rows]),
            "max_nodes": max(r["nodes"] for r in rows),
            "max_surviving": max(r["surviving"] for r in rows),
        })
        print("  d=%d: time exponent in n -- total %.2f, absorb %.2f, "
              "integrate %.2f" %
              (dim, summaries[-1]["exp_total"], summaries[-1]["exp_absorb"],
               summaries[-1]["exp_integrate"]))

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(args.out, "w") as handle:
        json.dump({"reference": REF, "records": records,
                   "summaries": summaries}, handle, indent=1, sort_keys=True)
    print("\nwrote %s (%d records)" % (args.out, len(records)))


if __name__ == "__main__":
    main()
