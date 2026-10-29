"""Command-line interface for stats-quick."""
import argparse
import csv
import json

from .core import describe, mann_whitney_u, one_sample_t_test, paired_t_test, student_t_test, welch_t_test


def read_column(path, column):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        out = []
        for row in csv.DictReader(fh):
            try:
                out.append(float(row[column]))
            except (ValueError, TypeError):
                pass
    return out


def read_groups(path, value, group):
    groups = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                groups.setdefault(row[group], []).append(float(row[value]))
            except (ValueError, TypeError):
                pass
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stats-quick", description="Quick statistics from a CSV file.")
    ap.add_argument("csvfile")
    ap.add_argument("-v", "--value", required=True, help="numeric column")
    ap.add_argument("-g", "--group", help="grouping column with exactly two groups")
    ap.add_argument("--paired", help="second numeric column for a paired t-test")
    ap.add_argument("--mu", type=float, help="one-sample t-test against this mean")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    res = {}
    if args.group:
        groups = read_groups(args.csvfile, args.value, args.group)
        if len(groups) != 2:
            ap.error("--group must have exactly 2 levels, found %d" % len(groups))
        (ga, x), (gb, y) = sorted(groups.items())
        res = {"groups": {ga: describe(x), gb: describe(y)}, "welch": welch_t_test(x, y),
               "student": student_t_test(x, y), "mann_whitney": mann_whitney_u(x, y)}
    else:
        x = read_column(args.csvfile, args.value)
        res["describe"] = describe(x)
        if args.paired:
            res["paired"] = paired_t_test(x, read_column(args.csvfile, args.paired))
        if args.mu is not None:
            res["one_sample"] = one_sample_t_test(x, args.mu)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    for key, val in res.items():
        if key == "groups":
            for g, d in val.items():
                print("%-12s n=%d mean=%.4g sd=%.4g median=%.4g" % (g, d["n"], d["mean"], d["sd"], d["median"]))
        elif key == "describe":
            print("n=%d mean=%.4g sd=%.4g median=%.4g IQR=[%.4g, %.4g] 95%% CI=[%.4g, %.4g]" % (
                val["n"], val["mean"], val["sd"], val["median"], val["q1"], val["q3"], *val.get("ci95", (0, 0))))
        elif "t" in val:
            print("%s: t(%.2f) = %.3f, p = %.4g, diff = %.4g, 95%% CI [%.4g, %.4g]" % (
                val["test"], val["df"], val["t"], val["p"], val["mean_diff"], *val["ci"]))
        else:
            print("%s: U = %.1f, z = %.3f, p = %.4g, rank-biserial r = %.3f" % (
                val["test"], val["U"], val["z"], val["p"], val["rank_biserial"]))
    return 0
