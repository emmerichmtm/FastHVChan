"""Peak memory of the Section-2 (d/2) solver, measured as live boxes.

The recursion is depth first and every node keeps its own clipped box array
alive while its first child runs, so the peak is the sum along a root-to-leaf
path, not the largest single node.  Chan's analysis says that sum is geometric
(the (d-2)-face weight drops by 2^(2/d) per level), hence linear; this script
checks that empirically by fitting the exponent in n.

    python benchmarks/space_profile.py [--quick] [--out FILE]

A box costs 2*dim doubles, so peak memory in words is 2*dim times the peak box
count reported here.
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
from crossbench import DATASETS, REF
from node_counts import fit_exponent


class SpaceMeter(ChanMeasure):
    """ChanMeasure that tracks live boxes and depth across the call stack."""

    def complement_measure(self, boxes, cell):
        self._live = 0
        self.peak_boxes = 0
        self.max_depth = 0
        return super().complement_measure(boxes, cell)

    def _measure(self, boxes, cell, depth):
        self._live += len(boxes)
        self.peak_boxes = max(self.peak_boxes, self._live)
        self.max_depth = max(self.max_depth, depth)
        try:
            return super()._measure(boxes, cell, depth)
        finally:
            self._live -= len(boxes)


# Sizes span a 4x-8x range per dimension so the fitted exponent is meaningful.
PLAN = {3: [50, 100, 200, 400, 800],
        4: [40, 80, 160, 320],
        5: [25, 50, 100, 200],
        6: [20, 40, 80, 160],
        7: [25, 50, 100]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="drop the largest size in each dimension")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--out", default=os.path.join(HERE, "results",
                                                      "space.json"))
    args = parser.parse_args()

    records, summaries = [], []
    print("%-10s %3s %6s %11s %8s %6s" %
          ("dataset", "d", "n", "peak boxes", "peak/n", "depth"))
    for dataset_name, generator in DATASETS:
        for dim, all_ns in sorted(PLAN.items()):
            ns = all_ns[:-1] if args.quick else all_ns
            peaks = []
            for n in ns:
                rng = random.Random(args.seed + 1000 * dim + n)
                points = generator(n, dim, rng)
                ref = tuple(REF for _ in range(dim))
                lo = tuple(min(p[k] for p in points) for k in range(dim))

                meter = SpaceMeter(dim)
                meter.complement_measure([Box(p, ref) for p in points],
                                         Box(lo, ref))
                peaks.append(meter.peak_boxes)
                records.append({
                    "dataset": dataset_name, "dim": dim, "n": n,
                    "peak_boxes": meter.peak_boxes,
                    "peak_words": meter.peak_boxes * 2 * dim,
                    "max_depth": meter.max_depth,
                    "nodes": meter.node_count,
                })
                print("%-10s %3d %6d %11d %8.2f %6d" %
                      (dataset_name, dim, n, meter.peak_boxes,
                       meter.peak_boxes / n, meter.max_depth))
                sys.stdout.flush()
            summaries.append({"dataset": dataset_name, "dim": dim,
                              "exponent": fit_exponent(ns, peaks),
                              "peak_over_n": peaks[-1] / ns[-1]})
            print("    fitted exponent in n: %.2f" % summaries[-1]["exponent"])

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(args.out, "w") as handle:
        json.dump({"seed": args.seed, "records": records,
                   "summaries": summaries}, handle, indent=1, sort_keys=True)
    print("\nwrote %s (%d records)" % (args.out, len(records)))


if __name__ == "__main__":
    main()
