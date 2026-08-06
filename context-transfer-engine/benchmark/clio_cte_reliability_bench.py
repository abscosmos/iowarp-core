#!/usr/bin/env python3
"""Driver for clio_cte_reliability_bench.

Runs the benchmark twice over the same dataset shape:

  1. COMBINED -- one run with an even split across all eight operations, so
     every operation competes with the others for runtime workers and storage.
  2. ISOLATED -- eight runs, each 100% of a single operation, with nothing else
     in flight.

Then prints the comparison: for the I/O operations, average bandwidth, IOPS and
p99 latency; for the metadata operations, average and p99 latency. The delta
between the two profiles is the interference cost -- how much an operation
degrades purely because the other seven are running alongside it.

The runtime and the clio_cte_filesystem pool must already be composed; this
driver only launches the benchmark binary.

Usage:
  clio_cte_reliability_bench.py --bin <path> [--duration 30] [--threads 8]
                               [--max-data 2g] [--root /rel_bench]
                               [--csv results.csv] [--only combined|isolated]
"""

import argparse
import csv
import os
import subprocess
import time
import sys

OPS = [
    "write_4k",
    "read_4k",
    "write_1m",
    "read_1m",
    "stat_size",
    "readdir_small",
    "readdir_large",
    "rename",
]

IO_OPS = {"write_4k", "read_4k", "write_1m", "read_1m"}


def run_one(args, label, mix, csv_path):
    """Invoke the benchmark `args.repeat` times, appending each run to the CSV.

    Repetition is not optional rigour here, it is the difference between a
    number and a guess. On a shared machine this workload drifted ~25% over
    tens of minutes and single runs disagreed by up to 2.3x, which is wide
    enough to manufacture a convincing 1.5x "improvement" out of nothing. Each
    repeat is written as its own row (label#N) so the report can take medians
    and show the spread instead of hiding it in one point estimate.
    """
    outputs = []
    for rep in range(args.repeat):
        rep_label = label if args.repeat == 1 else f"{label}#{rep + 1}"
        outputs.append(_run_once(args, rep_label, mix, csv_path))
        if rep + 1 < args.repeat and args.settle > 0:
            time.sleep(args.settle)
    return outputs[0]


def _run_once(args, label, mix, csv_path):
    """Invoke the benchmark once with the given mix, appending to csv_path."""
    cmd = [
        args.bin,
        "--threads", str(args.threads),
        "--duration", str(args.duration),
        "--warmup", str(args.warmup),
        "--max-data", args.max_data,
        "--root", args.root,
        "--label", label,
        "--mix", mix,
        "--csv", csv_path,
    ]
    print(f"[run] {label:<22} mix={mix}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=args.timeout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"benchmark failed for label={label} "
                         f"(rc={proc.returncode})")
    return proc.stdout


NUMERIC_FIELDS = ("ops_per_sec", "avg_us", "p50_us", "p99_us", "mb_per_sec",
                  "count", "errors", "bytes")


def _median(values):
    vs = sorted(values)
    n = len(vs)
    return vs[n // 2] if n % 2 else (vs[n // 2 - 1] + vs[n // 2]) / 2.0


def load(csv_path):
    """Read the benchmark CSV into {label: {op: row}}, collapsing repeats.

    Rows written by --repeat carry a "label#N" label. They are grouped back
    under the base label and reduced to the MEDIAN of each numeric field, with
    the observed min/max kept alongside so the report can state the spread.
    Median rather than mean: a single run that lands badly (this workload has
    produced 2.3x outliers on a shared machine) should not drag the summary.
    """
    groups = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            base = row["label"].split("#", 1)[0]
            groups.setdefault(base, {}).setdefault(
                row["operation"], []).append(row)

    out = {}
    for label, ops in groups.items():
        for op, rows in ops.items():
            merged = dict(rows[0])
            for field in NUMERIC_FIELDS:
                try:
                    vals = [float(r[field]) for r in rows]
                except (KeyError, ValueError):
                    continue
                merged[field] = _median(vals)
                merged[field + "_min"] = min(vals)
                merged[field + "_max"] = max(vals)
            merged["repeats"] = len(rows)
            out.setdefault(label, {})[op] = merged
    return out


def fmt(v, prec=2):
    return f"{float(v):,.{prec}f}"


def ratio(combined, isolated):
    """Isolated-relative change; >1 means combined is worse for latency."""
    try:
        c, i = float(combined), float(isolated)
        if i == 0:
            return "-"
        return f"{c / i:.2f}x"
    except (TypeError, ValueError):
        return "-"


def report(data):
    """Print the combined-vs-isolated comparison tables."""
    comb = data.get("combined", {})

    # Aggregate view first. Per-operation throughput in the combined run is
    # bounded by that operation's share of the mix (an even split gives each
    # op ~1/8 of the slots), so a per-op IOPS comparison against a 100%
    # isolated run is not like-for-like. The honest throughput question is how
    # many operations per second the system sustains in total, and the honest
    # per-op interference signal is latency.
    if comb:
        total_ops = sum(float(r["ops_per_sec"]) for r in comb.values())
        total_mb = sum(float(r["mb_per_sec"]) for r in comb.values())
        print()
        print(f"COMBINED aggregate: {total_ops:,.0f} ops/sec across all "
              f"operations, {total_mb:,.0f} MB/s total")
        best = max(((lab, next(iter(ops.values())))
                    for lab, ops in data.items()
                    if lab.startswith("isolated_")),
                   key=lambda kv: float(kv[1]["ops_per_sec"]), default=None)
        if best:
            print(f"Fastest isolated profile: {best[0].split('_', 1)[1]} at "
                  f"{float(best[1]['ops_per_sec']):,.0f} ops/sec")
        print("NOTE: each combined op gets ~1/N of the operation slots, so "
              "per-op IOPS below is mix-limited;")
        print("      latency (avg/p99) is the interference measure.")

    print()
    print("=" * 104)
    print("I/O OPERATIONS — combined mix vs isolated")
    print("=" * 104)
    hdr = (f"{'operation':<14}{'MB/s comb':>12}{'MB/s iso':>12}"
           f"{'IOPS comb':>12}{'IOPS iso':>12}"
           f"{'p99us comb':>13}{'p99us iso':>13}{'p99 ratio':>12}")
    print(hdr)
    print("-" * 104)
    for op in OPS:
        if op not in IO_OPS:
            continue
        c = comb.get(op)
        i = data.get(f"isolated_{op}", {}).get(op)
        if not c or not i:
            continue
        print(f"{op:<14}{fmt(c['mb_per_sec']):>12}{fmt(i['mb_per_sec']):>12}"
              f"{fmt(c['ops_per_sec']):>12}{fmt(i['ops_per_sec']):>12}"
              f"{fmt(c['p99_us']):>13}{fmt(i['p99_us']):>13}"
              f"{ratio(c['p99_us'], i['p99_us']):>12}")

    print()
    print("=" * 104)
    print("METADATA OPERATIONS — combined mix vs isolated")
    print("=" * 104)
    hdr = (f"{'operation':<14}{'avg_us comb':>14}{'avg_us iso':>14}"
           f"{'avg ratio':>12}{'p99us comb':>14}{'p99us iso':>14}"
           f"{'p99 ratio':>12}{'ops/s comb':>13}")
    print(hdr)
    print("-" * 104)
    for op in OPS:
        if op in IO_OPS:
            continue
        c = comb.get(op)
        i = data.get(f"isolated_{op}", {}).get(op)
        if not c or not i:
            continue
        print(f"{op:<14}{fmt(c['avg_us']):>14}{fmt(i['avg_us']):>14}"
              f"{ratio(c['avg_us'], i['avg_us']):>12}"
              f"{fmt(c['p99_us']):>14}{fmt(i['p99_us']):>14}"
              f"{ratio(c['p99_us'], i['p99_us']):>12}"
              f"{fmt(c['ops_per_sec']):>13}")

    # Spread across repeats. Printed unconditionally when repeats exist: a
    # median with a 2x range behind it is not a measurement, and the only way
    # to stop someone quoting it as one is to show the range next to it.
    reps = max((r.get("repeats", 1) for ops in data.values()
                for r in ops.values()), default=1)
    if reps > 1:
        print()
        print(f"RUN-TO-RUN SPREAD ({reps} runs per case, ops/sec)")
        print(f"{'label':<24}{'operation':<16}{'min':>12}{'median':>12}"
              f"{'max':>12}{'max/min':>10}")
        print("-" * 86)
        for lab in sorted(data):
            for op in OPS:
                r = data[lab].get(op)
                if not r or "ops_per_sec_min" not in r:
                    continue
                lo, hi = r["ops_per_sec_min"], r["ops_per_sec_max"]
                print(f"{lab:<24}{op:<16}{lo:>12,.0f}"
                      f"{float(r['ops_per_sec']):>12,.0f}{hi:>12,.0f}"
                      f"{(hi / lo if lo else 0):>9.2f}x")

    errs = [(lab, op, int(float(r["errors"]))) for lab, ops in data.items()
            for op, r in ops.items() if float(r["errors"]) > 0]
    if errs:
        print()
        print("FAILED OPERATIONS:")
        for lab, op, n in errs:
            print(f"  {lab}/{op}: {n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True, help="clio_cte_reliability_bench")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--warmup", type=float, default=2.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-data", default="2g")
    p.add_argument("--root", default="/rel_bench")
    p.add_argument("--csv", default="reliability_results.csv")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--repeat", type=int, default=3,
                   help="runs per case; the report takes the median and "
                        "shows the spread (default 3, 1 disables)")
    p.add_argument("--settle", type=float, default=10.0,
                   help="seconds between runs, so one run's shared memory is "
                        "reclaimed before the next allocates (default 10)")
    p.add_argument("--only", choices=["combined", "isolated", "report"],
                   help="run only one phase (default: all, then report)")
    args = p.parse_args()

    if args.only != "report" and os.path.exists(args.csv):
        os.remove(args.csv)

    even = 100.0 / len(OPS)
    if args.only in (None, "combined"):
        run_one(args, "combined",
                ",".join(f"{op}={even}" for op in OPS), args.csv)
        if args.only is None and args.settle > 0:
            time.sleep(args.settle)
    if args.only in (None, "isolated"):
        for idx, op in enumerate(OPS):
            run_one(args, f"isolated_{op}", f"{op}=100", args.csv)
            # Pause between phases. Each run hosts its own embedded runtime
            # with a multi-GB shared-memory segment; starting the next one
            # before the previous segment is reclaimed put the machine under
            # enough pressure that a run was SIGKILLed mid-suite. The pause is
            # cheap next to re-running a 9-case suite.
            if idx + 1 < len(OPS) and args.settle > 0:
                time.sleep(args.settle)

    report(load(args.csv))


if __name__ == "__main__":
    main()
