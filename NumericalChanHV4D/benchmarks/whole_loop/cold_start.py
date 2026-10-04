"""Measure one first compute with an empty Numba cache, then one warm compute.

Run as a new Python process. --cache-dir must be empty or not yet exist; this
script never removes cache files. Import time is measured separately, and the
first compute includes compilation, input conversion, and actual computation.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "cold_start.json")
    args = parser.parse_args()
    if "numba" in sys.modules:
        parser.error("Run this script in a new Python process, before importing Numba")
    cache = args.cache_dir.resolve()
    if cache.exists() and (not cache.is_dir() or any(cache.iterdir())):
        parser.error("--cache-dir must name an empty directory or a path that does not exist")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache)
    os.environ.pop("NUMBA_DISABLE_JIT", None)
    sys.path.insert(0, str(REPOSITORY))

    dataset = HERE.parent / "prefix4" / "datasets.json"
    case = next(case for case in json.loads(dataset.read_text(encoding="utf-8"))
                if case["id"] == "sphere-s42-n20")
    print(f"Empty-cache first-use measurement: {case['id']}", flush=True)
    started = time.perf_counter()
    from NumericalChanHV4D.numerical_chan4_compiled import NumericalChan4D, kernel
    import_seconds = time.perf_counter() - started

    def compute():
        started = time.perf_counter()
        engine = NumericalChan4D(base_hard=2)
        corners = [[r - x for r, x in zip(case["reference"], point)]
                   for point in case["points"]]
        value = engine.compute(corners)
        seconds = time.perf_counter() - started
        return {"seconds": seconds, "value": value, "counters": dict(engine.stats)}

    first = compute()
    print(f"First compute: {first['seconds']:.6f} seconds", flush=True)
    warm = compute()
    print(f"Warm compute: {warm['seconds']:.6f} seconds", flush=True)
    agree = math.isclose(first["value"], warm["value"], rel_tol=2e-9, abs_tol=2e-10)
    sources = [Path(__file__), dataset,
               REPOSITORY / "NumericalChanHV4D" / "numerical_chan4_compiled.py",
               REPOSITORY / "NumericalChanHV4D" / "compiled_terms.py",
               REPOSITORY / "NumericalChanHVND" / "numerical_chan_numba.py"]
    result = {
        "case": case["id"], "n": case["n"], "dimension": 4, "base_hard": 2,
        "cache_policy": "New process; empty dedicated NUMBA_CACHE_DIR before any Numba import; no files deleted",
        "cache_directory": str(cache), "cache_empty_before_import": True,
        "jit_enabled": True, "kernel_module": kernel.__name__,
        "import_seconds": import_seconds,
        "first_compute": first, "warm_compute": warm, "values_agree": agree,
        "timing_scope": "Each compute includes a fresh engine and r-p conversion; first compute includes compilation, not module import",
        "nopython_signatures": {name: len(getattr(getattr(kernel, name), "nopython_signatures", ()))
                                 for name in ("base_sum", "push_blocks", "apply_easy")},
        "cache_files_after": sum(path.is_file() for path in cache.rglob("*")),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "cpu": platform.processor(), "executable": sys.executable,
                        "numba": importlib.metadata.version("numba"),
                        "numpy": importlib.metadata.version("numpy")},
        "source_hashes": {str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return int(not agree)


if __name__ == "__main__":
    raise SystemExit(main())
