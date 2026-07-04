#!/usr/bin/env python3
"""
analyze_experts.py — turn moe_profile.csv into decisions.

Input CSV schema (produced by moe-profiler.h):  tag,layer,expert,count

What it computes, per workload tag:
  * total routed selections, layers seen, inferred n_expert
  * coverage: how many experts (per layer) are needed to cover
    50/80/90/95/99% of the routed mass  (mean and worst layer)
  * mass captured by the top {8,16,32,64,96} experts per layer (mean)
  * a verdict: is hot/cold tiering or expert pruning worth it?

Across tags:
  * mean per-layer Jaccard overlap of the top-K expert sets
    (low overlap => workload-specialized experts exist)

Usage:
  python3 analyze_experts.py moe_profile.csv [more.csv ...] [--top-k 32] [--plot]

stdlib only; --plot needs matplotlib.
"""

import argparse
import csv
import sys
from collections import defaultdict
from itertools import combinations


def load(paths):
    # tag -> layer -> expert -> count
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for p in paths:
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                data[row["tag"]][int(row["layer"])][int(row["expert"])] += int(row["count"])
    return data


def sorted_shares(counts):
    tot = sum(counts.values())
    if tot == 0:
        return []
    return sorted((c / tot for c in counts.values()), reverse=True)


def experts_for_mass(shares, p):
    cum = 0.0
    for i, s in enumerate(shares, 1):
        cum += s
        if cum >= p:
            return i
    return len(shares)


def mass_at_top(shares, k):
    return sum(shares[:k])


def top_set(counts, k):
    return set(e for e, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:k])


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def analyze_tag(tag, layers, top_ks, mass_targets):
    per_layer_shares = {il: sorted_shares(c) for il, c in layers.items() if c}
    n_expert = max((max(c) for c in layers.values() if c), default=-1) + 1
    total = sum(sum(c.values()) for c in layers.values())

    print(f"\n=== workload: {tag} ===")
    print(f"  layers: {len(layers)}   selections: {total:,}   inferred n_expert: {n_expert}")

    print(f"  experts needed per layer to cover routed mass (mean / worst layer):")
    for p in mass_targets:
        needs = [experts_for_mass(s, p) for s in per_layer_shares.values()]
        print(f"    {int(p*100):>3}% mass : {mean(needs):6.1f} / {max(needs):3d} experts")

    print(f"  mass captured by top-k experts per layer (mean over layers):")
    verdict_metric = None
    for k in top_ks:
        m = mean(mass_at_top(s, k) for s in per_layer_shares.values())
        print(f"    top {k:>3} : {m*100:5.1f}%")
        if n_expert > 0 and abs(k - n_expert // 4) <= max(2, n_expert // 16):
            verdict_metric = m  # mass at ~25% of experts

    if verdict_metric is None and per_layer_shares and n_expert > 0:
        verdict_metric = mean(mass_at_top(s, max(1, n_expert // 4)) for s in per_layer_shares.values())

    if verdict_metric is not None:
        if verdict_metric >= 0.90:
            v = "STRONG skew -> pruning / precision-tiering should pay off"
        elif verdict_metric >= 0.75:
            v = "MODERATE skew -> borderline; try pruning only the coldest tail"
        else:
            v = "FLAT routing -> skip tiering/pruning, buy memory instead"
        print(f"  verdict (mass at ~25% of experts = {verdict_metric*100:.1f}%): {v}")

    return per_layer_shares


def cross_tag(data, k):
    tags = sorted(data)
    if len(tags) < 2:
        return
    print(f"\n=== cross-workload overlap (top-{k} expert sets, per-layer Jaccard) ===")
    for a, b in combinations(tags, 2):
        common_layers = set(data[a]) & set(data[b])
        if not common_layers:
            continue
        js = []
        for il in common_layers:
            sa, sb = top_set(data[a][il], k), top_set(data[b][il], k)
            if sa or sb:
                js.append(len(sa & sb) / len(sa | sb))
        print(f"  {a:>12} vs {b:<12}: mean Jaccard = {mean(js):.2f}   "
              f"({'shared generalists' if mean(js) > 0.7 else 'workload-specialized experts' if mean(js) < 0.4 else 'mixed'})")


def plot(shares_by_tag, path="coverage_curves.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(--plot skipped: matplotlib not installed)", file=sys.stderr)
        return
    plt.figure(figsize=(7, 5))
    for tag, per_layer in shares_by_tag.items():
        n = max(len(s) for s in per_layer.values())
        curve = []
        for k in range(1, n + 1):
            curve.append(mean(mass_at_top(s, k) for s in per_layer.values()))
        plt.plot(range(1, n + 1), [c * 100 for c in curve], label=tag)
    plt.axhline(90, ls="--", lw=0.8, color="gray")
    plt.xlabel("top-k experts (per layer)")
    plt.ylabel("routed mass covered (%)")
    plt.title("MoE expert coverage curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="moe_profile.csv file(s)")
    ap.add_argument("--top-k", type=int, default=32, help="set size for cross-tag Jaccard")
    ap.add_argument("--plot", action="store_true", help="write coverage_curves.png")
    args = ap.parse_args()

    data = load(args.csvs)
    if not data:
        sys.exit("no rows found")

    top_ks = (8, 16, 32, 64, 96)
    mass_targets = (0.50, 0.80, 0.90, 0.95, 0.99)

    shares_by_tag = {}
    for tag in sorted(data):
        shares_by_tag[tag] = analyze_tag(tag, data[tag], top_ks, mass_targets)

    cross_tag(data, args.top_k)

    if args.plot:
        plot(shares_by_tag)


if __name__ == "__main__":
    main()
