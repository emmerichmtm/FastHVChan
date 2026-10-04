"""Repeatable 4D comparison; one fresh worker at a time, warm calls only.

Run with a Python environment containing NumPy and Numba::

    python benchmark_whole_loop.py --old /path/to/HV4DMagnitude

The seven inputs are the saved prefix4 audit inputs. Every implementation sees
the same union of boxes. The newer solvers receive upper corners r-p; the older
prefix solver and Chan references receive positive minimization points p and r.
Input conversion is timed, dominance prefiltering is disabled, and the new
engines retain their default compression schedule. A timeout applies to the
whole worker, including imports, an untimed first call, and all timed repeats.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
ALGORITHMS = ("whole_loop", "hybrid", "old_prefix", "chan_dby3", "chan_dby2")


def setup(algorithm, case, paths):
    old, new, reference = paths
    sys.path[:0] = [new, str(Path(new).parent), old, reference]
    os.environ.pop("NUMBA_DISABLE_JIT", None)
    points, ref = case["points"], tuple(case["reference"])
    if algorithm in ("whole_loop", "hybrid"):
        if algorithm == "whole_loop":
            from numerical_chan4_compiled import NumericalChan4D
            make_engine = lambda: NumericalChan4D(base_hard=2)
        else:
            from NumericalChanHVND.numerical_chan_numba import NumericalChanNumba
            make_engine = lambda: NumericalChanNumba(dimension=4, base_hard=2)

        def solve():
            engine = make_engine()
            corners = [[r - x for r, x in zip(ref, point)] for point in points]
            value = engine.compute(corners)
            return value, dict(engine.stats)
    elif algorithm == "old_prefix":
        from solver import PrefixHV4D

        def solve():
            engine = PrefixHV4D(base_hard=2, fast=True)
            return engine.hypervolume(points, ref), dict(engine.counters)
    elif algorithm == "chan_dby3":
        from chan_orthant_dby3 import ChanOrthantMeasure

        def solve():
            # Public hypervolume_dby3 adapter, retaining its engine's counters.
            front = [tuple(float(x) for x in point) for point in points
                     if all(x < r for x, r in zip(point, ref))]
            if not front:
                return 0.0, {}
            lo = tuple(min(point[i] for point in front) for i in range(4))
            engine = ChanOrthantMeasure(4, base_boxes=2)
            value = math.prod(r - l for l, r in zip(lo, ref))
            value -= engine.complement_measure(front, lo, ref)
            return value, {"nodes": engine.node_count,
                           "peak_terms": engine.term_high_water,
                           "compressions": engine.compress_count}
    else:
        from chan_hypervolume import hypervolume

        def solve():
            return hypervolume(points, ref, prefilter=False), {}
    return solve


def signatures(algorithm):
    """A nonzero count confirms compiled nopython execution was exercised."""
    if algorithm == "whole_loop":
        from numerical_chan4_compiled import kernel as compiled_terms
        return {name: len(getattr(getattr(compiled_terms, name),
                                  "nopython_signatures", ()))
                for name in ("base_sum", "push_blocks", "apply_easy")
                if hasattr(compiled_terms, name)}
    if algorithm == "old_prefix":
        import fastkernel
        return {"k_prefix": len(getattr(fastkernel.k_prefix,
                                       "nopython_signatures", ()))}
    return {}


def worker(pipe, algorithm, case, paths, repeats):
    try:
        solve = setup(algorithm, case, paths)
        start = time.perf_counter()
        value, counters = solve()
        first = {"first_seconds": time.perf_counter() - start,
                 "value": value, "counters": counters,
                 "nopython_signatures": signatures(algorithm)}
        pipe.send({"event": "first", **first})
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            value, counters = solve()
            samples.append({"seconds": time.perf_counter() - start,
                            "value": value, "counters": counters})
        elapsed = [sample["seconds"] for sample in samples]
        pipe.send({"event": "done", "status": "ok", **first,
                   "value": value, "counters": counters, "samples": samples,
                   "seconds": elapsed, "median_seconds": statistics.median(elapsed)})
    except Exception as error:
        pipe.send({"event": "done", "status": "error", "error": repr(error)})
    finally:
        pipe.close()


def isolated(algorithm, case, paths, repeats, timeout):
    reader, writer = mp.Pipe(duplex=False)
    process = mp.Process(target=worker, args=(writer, algorithm, case, paths, repeats))
    process.start()
    writer.close()
    deadline, first = time.monotonic() + timeout, {}
    try:
        while time.monotonic() < deadline:
            if not reader.poll(min(0.1, max(0.0, deadline - time.monotonic()))):
                continue
            try:
                result = reader.recv()
            except EOFError:
                process.join(2)
                return {**first, "status": "process_failed", "exit_code": process.exitcode}
            if result["event"] == "done":
                process.join(2)
                return result
            first = result
            # Do not stop on process.exitcode: a final result can still be in
            # the pipe behind the first-call message. Drain until done or EOF.
        return {**first, "status": "timeout", "budget_seconds": timeout}
    finally:
        if process.is_alive():
            process.terminate()
        process.join(3)
        reader.close()


def git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path),
             "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_hashes(old, new, reference, dataset):
    files = {"harness": Path(__file__), "datasets": dataset}
    for label, folder, names in (
        ("old", old, ("solver.py", "fastkernel.py", "prefix4.py", "core.py")),
        ("whole_loop", new, ("numerical_chan4_compiled.py", "compiled_terms.py")),
        ("hybrid", new.parent / "NumericalChanHVND",
         ("numerical_chan.py", "numerical_chan_numba.py")),
        ("reference", reference, ("chan_orthant_dby3.py", "chan_hypervolume.py")),
    ):
        files.update({f"{label}/{name}": folder / name for name in names})
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in files.items()}


def save(report, out):
    (out / "timings.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, extrasaction="ignore", fieldnames=(
            "id", "family", "seed", "n", "algorithm", "status", "value",
            "first_seconds", "median_seconds", "scaled_error"))
        writer.writeheader()
        writer.writerows(report["rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, default=REPOSITORY / "NumericalChanHV4D")
    parser.add_argument("--reference", type=Path, default=REPOSITORY)
    parser.add_argument("--datasets", type=Path, default=HERE.parent / "prefix4" / "datasets.json")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--only", default="", help="Keep case IDs containing this text")
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    old, new, reference = (path.resolve() for path in (args.old, args.new, args.reference))
    cases = [case for case in json.loads(args.datasets.read_text(encoding="utf-8"))
             if args.only in case["id"]]
    if not cases:
        parser.error("No datasets match --only")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "datasets.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    report = {
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "cpu": platform.processor(), "executable": sys.executable,
                        "numba": importlib.metadata.version("numba"),
                        "numpy": importlib.metadata.version("numpy")},
        "method": "Sequential fresh workers; untimed first call then warm end-to-end repeats",
        "dimension": 4, "base_hard": 2, "prefilter": False,
        "compression_schedule": "default, unchanged",
        "repeats": args.repeats, "worker_timeout_seconds": args.timeout,
        "source_revisions": {"old": git_revision(old), "reference": git_revision(reference)},
        "source_hashes": source_hashes(old, new, reference, args.datasets),
        "rows": [],
    }
    paths = tuple(str(path) for path in (old, new, reference))
    errors = 0
    for case in cases:
        order = list(ALGORITHMS)
        random.Random(case["id"]).shuffle(order)
        current = []
        for algorithm in order:
            row = {key: value for key, value in case.items() if key not in ("points", "reference")}
            row.update(algorithm=algorithm)
            row.update(isolated(algorithm, case, paths, args.repeats, args.timeout))
            current.append(row)
            report["rows"].append(row)
            save(report, args.out)
            print(case["id"], algorithm, row["status"], row.get("median_seconds", ""), flush=True)
        baseline = next(row for row in current if row["algorithm"] == "chan_dby2")
        for row in current:
            if row["status"] == "error" or row["status"] == "process_failed":
                errors += 1
            if row["status"] != "ok":
                continue
            if baseline["status"] != "ok":
                row["validation"] = "unavailable: Chan d/2 baseline did not complete"
                errors += 1
                continue
            expected = baseline["value"]
            values = [row["value"]] + [sample["value"] for sample in row["samples"]]
            row["scaled_error"] = max(abs(value - expected) / max(1.0, abs(expected))
                                      for value in values)
            row["validation"] = "agrees with Chan d/2"
            if not all(math.isclose(value, expected, rel_tol=2e-9, abs_tol=2e-10) for value in values):
                row["status"], row["validation"] = "value_mismatch", "failed"
                errors += 1
        save(report, args.out)
    report["validation_errors"] = errors
    save(report, args.out)
    return int(errors != 0)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
