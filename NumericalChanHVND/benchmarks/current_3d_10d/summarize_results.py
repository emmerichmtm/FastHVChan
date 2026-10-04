"""Render the saved 3D--10D benchmark records without rerunning a solver.

Run after benchmark_dimensions.py; an interrupted run is labeled partial.
Only completed, successful pairs contribute to reported timing ratios.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
MAIN_ALGORITHMS = ("numerical_python", "current_numba", "chan_dby3", "chan_dby2")
NAMES = {
    "numerical_python": "Numerical Python",
    "current_numba": "Current Numba",
    "chan_dby3": "Chan d/3 Python",
    "chan_dby2": "Chan d/2 Python",
    "hybrid_4d": "Previous 4D hybrid",
}


def good(row):
    return row is not None and row.get("status") == "ok"


def timing(row):
    if row is None:
        return "pending"
    if good(row):
        return f"{1000 * row['median_seconds']:.3f}"
    return row.get("status", "unknown").replace("_", " ")


def error_count(value):
    return len(value) if isinstance(value, (list, dict)) else int(value or 0)


def status_text(rows):
    counts = Counter(row.get("status", "unknown") for row in rows)
    return ", ".join(f"{count} {status.replace('_', ' ')}"
                     for status, count in sorted(counts.items())) or "none recorded"


def paired_ratios(cases, indexed, numerator, denominator):
    values = []
    for case in cases:
        a = indexed.get((case["id"], numerator))
        b = indexed.get((case["id"], denominator))
        if good(a) and good(b) and b["median_seconds"] > 0:
            values.append(a["median_seconds"] / b["median_seconds"])
    return values


def ratio_cell(values):
    return f"{statistics.median(values):.3g}x ({len(values)})" if values else "unavailable"


def key_findings(cases, indexed):
    ordinary = [case for case in cases if not case["magnitude"]]
    lines = []
    sweep = paired_ratios([case for case in ordinary if case["d"] == 3], indexed,
                          "numerical_python", "current_numba")
    if sweep:
        median = statistics.median(sweep)
        lines.append(f"The 3D compiled sweep has a median Python-time/Numba-time ratio of "
                     f"{median:.2f}x over {len(sweep)} completed ordinary-HV pairs; Numba is "
                     f"faster in {sum(value > 1 for value in sweep)} of those pairs.")
    whole = paired_ratios([case for case in ordinary if case["d"] == 4], indexed,
                          "hybrid_4d", "current_numba")
    if whole:
        lines.append(f"The 4D whole-contraction backend has a median previous-hybrid/current "
                     f"time ratio of {statistics.median(whole):.2f}x over {len(whole)} completed "
                     f"ordinary-HV pairs (range {min(whole):.2f}–{max(whole):.2f}x).")
    higher = []
    for dimension in range(5, 11):
        ratios = paired_ratios([case for case in ordinary if case["d"] == dimension],
                               indexed, "current_numba", "numerical_python")
        if ratios:
            higher.append(f"{dimension}D: {statistics.median(ratios):.2f}x ({len(ratios)} pairs)")
    if higher:
        lines.append("For the higher-dimensional hybrid, median Numba-time/Python-time "
                     "ratios within each dimension are " + "; ".join(higher) +
                     ". Here ratios above one mean the hybrid is slower than numerical Python.")
    reference_cases = [case for case in ordinary if case["d"] >= 4
                       and good(indexed.get((case["id"], "chan_dby2")))]
    reference_wins = sum(
        indexed[case["id"], "chan_dby2"]["median_seconds"] <= min(
            indexed[case["id"], name]["median_seconds"] for name in MAIN_ALGORITHMS
            if good(indexed.get((case["id"], name))))
        for case in reference_cases)
    if reference_cases:
        lines.append(f"The Python Chan d/2 reference has the lowest completed median on "
                     f"{reference_wins} of {len(reference_cases)} ordinary-HV cases in 4D-10D.")
    if lines:
        lines.append("These completed-pair summaries have selection bias: the slowest cases "
                     "that time out are excluded. They do not describe the cost or success "
                     "rate of the full workload; the status counts must be read alongside them.")
    return lines


def oracle_summary(oracle, cases):
    checked = oracle["cases"]
    if oracle["checked_cases"] != len(checked):
        raise ValueError("Exact-oracle checked_cases differs from its case records.")
    selected = {case["id"] for case in cases if case["n"] <= oracle["max_points"]}
    checked_ids = [case["id"] for case in checked]
    if len(checked_ids) != len(set(checked_ids)) or set(checked_ids) != selected:
        raise ValueError("Exact-oracle cases do not match the selected saved inputs.")
    return (f"The independent [exact rational inclusion–exclusion audit]"
            f"(results/oracle_verification.json) reports **{oracle['status']}**: "
            f"**{oracle['checked_cases']} inputs** with at most {oracle['max_points']} points, "
            f"**{oracle['checked_values']} recorded values**, and maximum scaled error "
            f"**{oracle['max_scaled_error']:.3g}**. The case count is checked against the "
            "saved input selection. The oracle uses the exact rational values of the binary64 "
            "input coordinates and shares no solver or compression code. Its value checks "
            "include any available samples from timed-out workers; they do not turn those "
            "workers into completed timing measurements.")


def render(report, cases, oracle=None):
    rows = report["rows"]
    indexed = {(row["id"], row["algorithm"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate case/algorithm rows: refusing ambiguous summary.")
    expected = sum(4 + (case["d"] == 4) for case in cases)
    complete = len(rows) == expected and bool(report.get("finished_utc"))
    successful = [row for row in rows if good(row)]
    errors = error_count(report["validation_errors"]) if "validation_errors" in report else "pending"
    hv_cases = [case for case in cases if not case["magnitude"]]
    mag_cases = [case for case in cases if case["magnitude"]]
    env = report.get("environment", {})
    budget = report.get("worker_timeout_seconds", report.get("timeout_seconds", "recorded"))
    repeats = report.get("repeats", "recorded")
    out = ["# Current numerical hypervolume benchmarks, 3D–10D", ""]

    if not complete:
        out += [f"**Partial run:** {len(rows)} of {expected} planned workers are recorded. "
                "Pending cells are not failures or measured timings.", ""]
    out += [f"This run records {len(cases)} inputs and {len(rows)} algorithm/input workers: "
            f"{status_text(rows)}. Validation errors recorded by the runner: **{errors}**.", "",
            "The benchmark selects a compiled sweep for its current Numba row in 3D, the new whole-contraction "
            "backend with persistent native term states in 4D, and the existing hybrid backend "
            "in 5D–10D. Geometric recursion remains Python in 4D–10D. This benchmark does not "
            "extend the whole-contraction implementation to higher dimensions.", ""]
    findings = key_findings(cases, indexed)
    if findings:
        out += ["## Key findings", ""]
        for finding in findings:
            out += [finding, ""]
    out += ["## Method", "",
            f"One fresh worker runs at a time. Each implementation receives one untimed first "
            f"call followed by {repeats} timed calls; tables report their median in milliseconds. "
            "The timed region includes input conversion and rank preprocessing, but excludes "
            "module imports. Every call constructs a fresh solver. The default compression "
            "schedule is retained and the Chan dominance prefilter is disabled. No fastmath "
            "or parallel numerical execution is used.", "",
            f"The {budget}-second limit applies to the complete worker, including startup, "
            "imports, the first call, and all timed repetitions. A timeout therefore gives "
            "no per-call runtime lower bound. No ratio is calculated from a timeout.", "",
            "All implementations receive the same saved box union. Numerical solvers receive "
            "nonnegative anchored extents; the two Python Chan references receive the equivalent "
            "minimization representation. For magnitude the references receive the affine HV "
            "transformation `1 + extent/2`, with the transformation included in timing. "
            "The sphere inputs have mutually nondominated corners; the tied-grid inputs can "
            "contain dominated and duplicate points. Inputs are not filtered before timing.", "",
            "The 3D sweep and the reference Chan algorithms differ algorithmically, so their "
            "timing ratio combines algorithm and implementation effects. Numerical Python "
            "versus Numba is the closer backend comparison. These are small, dimension-dependent "
            "test sizes; the timings do not estimate an asymptotic exponent or guarantee "
            "performance on large high-dimensional fronts.", ""]

    version = str(env.get("python", "not recorded")).splitlines()[0]
    out += [f"Environment: Python `{version}`; NumPy `{env.get('numpy', 'not recorded')}`; "
            f"Numba `{env.get('numba', 'not recorded')}`; `{env.get('platform', 'not recorded')}`; "
            f"CPU `{env.get('cpu', 'not recorded')}`.", "",
            "## Ordinary hypervolume", "",
            "All times below are milliseconds. Status words denote unfinished or failed "
            "workers rather than numeric measurements.", "",
            "| Input | Numerical Python | Current Numba | Chan d/3 Python | Chan d/2 Python |",
            "|---|---:|---:|---:|---:|"]
    for case in hv_cases:
        values = [timing(indexed.get((case["id"], name))) for name in MAIN_ALGORITHMS]
        out.append(f"| {case['id']} | " + " | ".join(values) + " |")

    out += ["", "## Dimension-by-dimension comparison", "",
            "Ratios are baseline time divided by current Numba time, using only completed "
            "ordinary-HV pairs. A ratio above one means Numba was faster. Each cell reports "
            "the median of per-input ratios, followed by the number of matched inputs. "
            "Changing the dimension also changes the tested point counts; compare within "
            "a dimension rather than reading this table as a scaling curve.", "",
            "| Dimension | Current backend | Current worker statuses | Python numerical / Numba | Chan d/3 / Numba | Chan d/2 / Numba |",
            "|---|---|---|---:|---:|---:|"]
    for d in range(3, 11):
        subset = [case for case in hv_cases if case["d"] == d]
        if not subset:
            continue
        cur_rows = [indexed[(case["id"], "current_numba")] for case in subset
                    if (case["id"], "current_numba") in indexed]
        backend = report.get("backend_by_dimension", {}).get(str(d),
            "compiled sweep" if d == 3 else "whole contractions" if d == 4 else "hybrid")
        ratios = [ratio_cell(paired_ratios(subset, indexed, name, "current_numba"))
                  for name in ("numerical_python", "chan_dby3", "chan_dby2")]
        out.append(f"| {d}D | {backend} | {status_text(cur_rows)} | " + " | ".join(ratios) + " |")

    out += ["", "## 4D check against the previous hybrid", "",
            "This compares the new 4D backend with the generic ND hybrid on identical inputs. "
            "The older HV4DMagnitude prefix solver is outside this 3D–10D comparison; its "
            "earlier dedicated 4D comparison remains available "
            "[here](../../../NumericalChanHV4D/benchmarks/whole_loop/RESULTS.md).", "",
            "| Input | Current whole-contraction Numba (ms) | Previous hybrid (ms) | Hybrid / current |",
            "|---|---:|---:|---:|"]
    for case in [case for case in cases if case["d"] == 4]:
        ratio = paired_ratios([case], indexed, "hybrid_4d", "current_numba")
        ratio_text = f"{ratio[0]:.2f}x" if ratio else "unavailable"
        out.append(f"| {case['id']} | {timing(indexed.get((case['id'], 'current_numba')))} | "
                   f"{timing(indexed.get((case['id'], 'hybrid_4d')))} | {ratio_text} |")

    out += ["", "## Magnitude", "",
            "These additional sphere cases compare direct numerical magnitude with its "
            "equivalent affine-transformed HV in both Chan references. Times are milliseconds. "
            "They are excluded from the ordinary-HV ratio table above.", "",
            "| Input | Numerical Python | Current Numba | Chan d/3 Python | Chan d/2 Python |",
            "|---|---:|---:|---:|---:|"]
    for case in mag_cases:
        values = [timing(indexed.get((case["id"], name))) for name in MAIN_ALGORITHMS]
        out.append(f"| {case['id']} | " + " | ".join(values) + " |")

    out += ["", "## First use and validation", "",
            "The following ranges include every recorded first-call duration for current "
            "Numba, including a first call whose subsequent repetitions exceeded the worker "
            "budget. A first call may compile specializations or load existing cached code; "
            "these are **not fresh-cache compilation measurements**. The first-call "
            "durations exclude imports and include computation. They are not included in "
            "the warm medians.", "",
            "| Dimension | Recorded first calls | First-call range (s) |",
            "|---|---:|---:|"]
    for d in range(3, 11):
        values = [row["first_seconds"] for row in rows
                  if row.get("d") == d and row["algorithm"] == "current_numba"
                  and "first_seconds" in row]
        if values:
            out.append(f"| {d}D | {len(values)} | {min(values):.4f}–{max(values):.4f} |")
        else:
            out.append(f"| {d}D | 0 | unavailable |")

    checked = [row for row in successful if "scaled_error" in row]
    if checked:
        maximum = max(row["scaled_error"] for row in checked)
        out += ["", f"Across {len(checked)} completed rows with a recorded comparison, the largest "
                f"scaled disagreement with Chan d/2 was **{maximum:.3g}**. Scaled error is "
                "absolute disagreement divided by `max(1, abs(reference value))`. The runner "
                "checks the first call and all timed repetitions; incomplete workers are not "
                "counted as correctness passes. Agreement is a numerical cross-check, not "
                "a replacement for the existing exact-oracle test suites."]
    else:
        out += ["", "No completed rows with recorded numerical comparisons are available yet."]

    if oracle is not None:
        out += ["", oracle_summary(oracle, cases)]

    out += ["", "## Reproduction records", "",
            "The saved [input arrays](results/datasets.json), [full timing and counter records]"
            "(results/timings.json), and [CSV timings](results/timings.csv) preserve the cases, "
            "per-call values, statuses, source hashes, and environment. "
            "Additional [hardware details](results/hardware.json) identify the measured host. "
            "[benchmark_dimensions.py](benchmark_dimensions.py) runs the experiment; "
            "[summarize_results.py](summarize_results.py) builds this document from those records "
            "without rerunning an algorithm.", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, default=HERE / "RESULTS.md")
    args = parser.parse_args()
    report = json.loads((args.results_dir / "timings.json").read_text(encoding="utf-8"))
    cases = json.loads((args.results_dir / "datasets.json").read_text(encoding="utf-8"))
    oracle_file = args.results_dir / "oracle_verification.json"
    oracle = json.loads(oracle_file.read_text(encoding="utf-8")) if oracle_file.exists() else None
    args.out.write_text(render(report, cases, oracle), encoding="utf-8", newline="\n")
    print(args.out)


if __name__ == "__main__":
    main()
