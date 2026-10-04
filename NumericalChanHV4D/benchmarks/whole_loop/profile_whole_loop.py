"""Profile one warm 4D call, separately from benchmark timing workers."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats

from benchmark_whole_loop import HERE, REPOSITORY, setup, signatures, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", type=Path, default=REPOSITORY / "NumericalChanHV4D")
    parser.add_argument("--reference", type=Path, default=REPOSITORY)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "profile.json")
    parser.add_argument("--case", default="sphere-s42-n20")
    args = parser.parse_args()
    old, new, reference = (path.resolve() for path in (args.old, args.new, args.reference))
    dataset = HERE.parent / "prefix4" / "datasets.json"
    case = next(case for case in json.loads(dataset.read_text(encoding="utf-8"))
                if case["id"] == args.case)
    solve = setup("whole_loop", case, tuple(str(path) for path in (old, new, reference)))
    solve()  # Compilation/cache loading excluded from the profile.
    profiler = cProfile.Profile()
    profiler.enable()
    value, counters = solve()
    profiler.disable()
    stats = pstats.Stats(profiler)
    functions = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        functions.append({"file": filename, "line": line, "function": name,
                          "primitive_calls": primitive, "calls": calls,
                          "self_seconds": own, "cumulative_seconds": cumulative})
    result = {"case": case["id"], "algorithm": "whole_loop", "value": value,
              "counters": counters, "total_seconds": stats.total_tt,
              "total_calls": stats.total_calls, "nopython_signatures": signatures("whole_loop"),
              "source_hashes": source_hashes(old, new, reference, dataset),
              "by_self": sorted(functions, key=lambda item: item["self_seconds"], reverse=True)[:80],
              "by_cumulative": sorted(functions, key=lambda item: item["cumulative_seconds"], reverse=True)[:80]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("case", "total_seconds", "total_calls", "counters")}, indent=2))
    for item in result["by_cumulative"][:20]:
        print(f'{item["cumulative_seconds"]:.6f} cumulative, {item["self_seconds"]:.6f} self, '
              f'{item["calls"]} calls: {Path(item["file"]).name}:{item["line"]} {item["function"]}')


if __name__ == "__main__":
    main()
