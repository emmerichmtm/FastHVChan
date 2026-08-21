"""Cross-implementation benchmark: Python vs C, several variants, up to d=10.

Every variant sees byte-identical input, written to a dataset file first, so
the timings are comparable and the returned hypervolumes can be cross-checked
against each other.  Each measurement runs in its own subprocess under a wall
clock limit; a variant that exceeds it is recorded as a timeout and is not
tried again at a larger size for that dimension.

    python benchmarks/crossbench.py [--timeout S] [--quick] [--out FILE]

Writes JSON that benchmarks/make_bench_tex.py turns into the LaTeX report.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

REF = 1.2  # reference-point coordinate, as in benchmarks/benchmark.py
RADIUS = 1.0

# (name, kind, extra argv) -- kind is "py" (worker script) or "c" (driver).
VARIANTS = [
    ("py-sec2", "py", []),
    ("py-sec2-nopre", "py", []),
    ("py-dby3", "py", []),
    ("py-dby3-nocomp", "py", []),
    ("py-ds", "py", []),
    ("py-wfg", "py", []),
    ("py-ys", "py", []),
    ("c-sec2", "c", []),
    ("c-sec2-bc8", "c", ["--base-case", "8"]),
    ("c-ds", "c", ["--algo", "ds"]),
    ("c-wfg", "c", ["--algo", "wfg"]),
]

DIMS = [3, 4, 5, 7, 9, 10]

SIZES = {
    3: [100, 200],
    4: [60, 120],
    5: [40, 80],
    7: [25, 50],
    9: [20, 40],
    10: [20, 40],
}

QUICK_SIZES = {d: sizes[:1] for d, sizes in SIZES.items()}


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def spherical(n, dim, rng):
    """Points on the unit sphere in the positive orthant.

    Every point is mutually non-dominated in all `dim` objectives -- the
    standard hard case, since dominance pruning removes nothing.
    """
    points = []
    for _ in range(n):
        coordinates = [rng.random() + 1e-12 for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in coordinates))
        points.append(tuple(RADIUS * x / norm for x in coordinates))
    return points


def cliff(n, dim, rng):
    """Non-dominated in the first two objectives, random in the rest.

    The first two coordinates lie on a quarter-circle arc, so no point can
    dominate another whatever the remaining coordinates do: the entire
    dominance structure sits in a 2-D slice while the other d-2 objectives are
    pure noise.  The front is as large as in the spherical case, but its
    geometry is degenerate in a way the recursion has to discover.
    """
    points = []
    for i in range(n):
        angle = (i + 0.5) / n * (math.pi / 2)
        head = (RADIUS * math.cos(angle), RADIUS * math.sin(angle))
        tail = tuple(rng.random() for _ in range(dim - 2))
        points.append(head + tail)
    rng.shuffle(points)
    return points


DATASETS = [("spherical", spherical), ("cliff", cliff)]


def write_dataset(path, points, dim):
    ref = [REF] * dim
    with open(path, "w") as handle:
        handle.write("%d %d\n" % (dim, len(points)))
        handle.write(" ".join(repr(float(x)) for x in ref) + "\n")
        for point in points:
            handle.write(" ".join(repr(float(x)) for x in point) + "\n")


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def parse_kv(text):
    out = {}
    for token in text.split():
        if "=" in token:
            key, _, value = token.partition("=")
            out[key] = value
    return out


def measure(variant, kind, extra, path, driver, timeout):
    if kind == "py":
        argv = [sys.executable, os.path.join(HERE, "crossbench_worker.py"),
                variant, path]
    else:
        argv = [driver, path] + extra
    try:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    if done.returncode != 0:
        return {"status": "error",
                "detail": (done.stderr or done.stdout).strip()[:200]}
    fields = parse_kv(done.stdout)
    if "seconds" not in fields or "hv" not in fields:
        return {"status": "error", "detail": done.stdout.strip()[:200]}
    result = {
        "status": "ok",
        "seconds": float(fields["seconds"]),
        "hv": float(fields["hv"]),
        "reps": int(fields.get("reps", 1)),
    }
    if "nodes" in fields:
        result["nodes"] = int(fields["nodes"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="wall clock limit per measurement, seconds")
    parser.add_argument("--quick", action="store_true",
                        help="smallest size per dimension only")
    parser.add_argument("--out", default=os.path.join(HERE, "results",
                                                      "crossbench.json"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    driver = os.path.join(ROOT, "c",
                          "bench_driver.exe" if os.name == "nt" else "bench_driver")
    variants = VARIANTS
    if not os.path.exists(driver):
        variants = [v for v in VARIANTS if v[1] != "c"]
        print("WARNING: C driver not found (%s); skipping C variants.\n"
              "Build it with:  make -C c bench_driver" % driver)

    data_dir = args.data_dir or tempfile.mkdtemp(prefix="crossbench-")
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    sizes = QUICK_SIZES if args.quick else SIZES

    records = []
    # A variant that has already blown the budget at some size is not retried
    # at a larger one -- these curves are monotone in n.
    exhausted = set()

    for dataset_name, generator in DATASETS:
        for dim in DIMS:
            for n in sizes[dim]:
                rng = random.Random(args.seed + 1000 * dim + n)
                points = generator(n, dim, rng)
                path = os.path.join(data_dir,
                                    "%s-d%d-n%d.txt" % (dataset_name, dim, n))
                write_dataset(path, points, dim)

                for variant, kind, extra in variants:
                    key = (dataset_name, dim, variant)
                    if key in exhausted:
                        outcome = {"status": "skipped"}
                    else:
                        outcome = measure(variant, kind, extra, path, driver,
                                          args.timeout)
                        if outcome["status"] in ("timeout", "error"):
                            exhausted.add(key)
                    record = {"dataset": dataset_name, "dim": dim, "n": n,
                              "variant": variant}
                    record.update(outcome)
                    records.append(record)

                    status = outcome["status"]
                    detail = ("%.4f s" % outcome["seconds"]
                              if status == "ok" else status)
                    print("%-9s d=%-2d n=%-4d %-16s %s"
                          % (dataset_name, dim, n, variant, detail))
                    sys.stdout.flush()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    payload = {
        "reference": REF,
        "seed": args.seed,
        "timeout": args.timeout,
        "dims": DIMS,
        "sizes": {str(k): v for k, v in sizes.items()},
        "variants": [v[0] for v in variants],
        "records": records,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print("\nwrote %s (%d records)" % (args.out, len(records)))


if __name__ == "__main__":
    main()
