"""Run one timed measurement and print it as key=value pairs.

Invoked as a subprocess by crossbench.py so that a variant which takes too
long can be killed without taking the whole run down with it.  Timing covers
the call only: the dataset is parsed first, outside the timed region.

    python crossbench_worker.py <variant> <dataset-file>
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_SECONDS = 0.05  # repeat fast cases so clock granularity does not dominate
MAX_REPS = 200


def read_dataset(path):
    with open(path) as handle:
        tokens = handle.read().split()
    dim = int(tokens[0])
    n = int(tokens[1])
    values = [float(x) for x in tokens[2:]]
    ref = tuple(values[:dim])
    flat = values[dim:]
    points = [tuple(flat[i * dim:(i + 1) * dim]) for i in range(n)]
    return dim, points, ref


def make_call(variant, points, ref):
    if variant == "py-sec2":
        from chan_hypervolume import hypervolume
        return lambda: hypervolume(points, ref)
    if variant == "py-sec2-nopre":
        from chan_hypervolume import hypervolume
        return lambda: hypervolume(points, ref, prefilter=False)
    if variant == "py-dby3":
        from chan_orthant_dby3 import hypervolume_dby3
        return lambda: hypervolume_dby3(points, ref)
    if variant == "py-dby3-nocomp":
        from chan_orthant_dby3 import hypervolume_dby3
        return lambda: hypervolume_dby3(points, ref, use_compression=False)
    if variant == "py-ds":
        from hv_baselines import hypervolume_ds
        return lambda: hypervolume_ds(points, ref)
    if variant == "py-wfg":
        from hv_baselines import hypervolume_wfg
        return lambda: hypervolume_wfg(points, ref)
    if variant == "py-ys":
        # Vendored Yildiz-Suri-style anchored solver; a minimisation instance
        # maps to the anchored form through q = ref - y.  The transform is part
        # of the timed call, like the other variants' prefilters.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from vendor.yildiz_suri_anchor import anchored_hypervolume

        def call():
            anchored = [
                tuple(r - x for x, r in zip(p, ref))
                for p in points
                if all(x < r for x, r in zip(p, ref))
            ]
            return anchored_hypervolume(anchored, method="auto")

        return call
    raise SystemExit("unknown variant %r" % variant)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    variant, path = sys.argv[1], sys.argv[2]
    _dim, points, ref = read_dataset(path)
    call = make_call(variant, points, ref)

    value = None
    reps = 0
    start = time.perf_counter()
    while True:
        value = call()
        reps += 1
        elapsed = time.perf_counter() - start
        if elapsed >= MIN_SECONDS or reps >= MAX_REPS:
            break

    print("hv=%.17g seconds=%.9f reps=%d" % (value, elapsed / reps, reps))


if __name__ == "__main__":
    main()
