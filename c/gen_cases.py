"""Generate differential-test cases from the Python FastHVChan implementation.

Writes cases plus the values the Python code produces; test_fasthvchan --check
replays them through the C port and compares.  Point the environment variable
FASTHVCHAN_PATH at the Python checkout if it is not the parent directory.

    python gen_cases.py cases.txt [count]
"""

import os
import random
import sys

_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.environ.get("FASTHVCHAN_PATH", _DEFAULT)))

from chan_hypervolume import hypervolume, union_volume


def fmt(x):
    return repr(float(x))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "cases.txt"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rng = random.Random(20260821)
    lines = []

    for _ in range(count):
        dim = rng.choice([2, 2, 3, 3, 3, 4, 4, 5, 6])
        n = rng.choice([1, 2, 3, 5, 8, 13, 21, 34])
        maximize = rng.random() < 0.25
        prefilter = rng.random() < 0.75

        # A mix of continuous coordinates and a coarse grid, so ties, shared
        # faces and duplicate points all get exercised, not just generic input.
        if rng.random() < 0.35:
            grid = rng.choice([2, 3, 4])
            pts = [
                tuple(rng.randint(0, grid) / grid for _ in range(dim))
                for _ in range(n)
            ]
        else:
            pts = [tuple(rng.random() for _ in range(dim)) for _ in range(n)]

        if maximize:
            ref = tuple(-rng.random() * 0.2 for _ in range(dim))
            pts = [tuple(x for x in p) for p in pts]
        else:
            ref = tuple(1.0 + rng.random() * 0.2 for _ in range(dim))

        expected = hypervolume(pts, ref, maximize=maximize, prefilter=prefilter)
        flat = [x for p in pts for x in p]
        lines.append(
            "hv %d %d %d %d %s %s %s"
            % (
                dim,
                n,
                1 if maximize else 0,
                1 if prefilter else 0,
                fmt(expected),
                " ".join(fmt(x) for x in ref),
                " ".join(fmt(x) for x in flat),
            )
        )

    for _ in range(count // 2):
        dim = rng.choice([2, 2, 3, 3, 4])
        n = rng.choice([1, 2, 3, 5, 8, 13])
        boxes = []
        for _ in range(n):
            lo, hi = [], []
            for _ in range(dim):
                if rng.random() < 0.3:  # snap to a grid to force shared faces
                    a, b = rng.randint(0, 4) / 4.0, rng.randint(0, 4) / 4.0
                else:
                    a, b = rng.random(), rng.random()
                lo.append(min(a, b))
                hi.append(max(a, b))
            boxes.append((tuple(lo), tuple(hi)))
        expected = union_volume(boxes)
        flat = []
        for lo, hi in boxes:
            flat.extend(lo)
            flat.extend(hi)
        lines.append(
            "uv %d %d %s %s"
            % (dim, n, fmt(expected), " ".join(fmt(x) for x in flat))
        )

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote %d cases to %s" % (len(lines), path))


if __name__ == "__main__":
    main()
