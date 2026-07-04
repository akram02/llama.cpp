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


def hot_table(tag, layers, n_top):
    """Per-layer top-N expert table, like:
    ┌───────┬───────┬─────────────────────────────┐
    │ layer │ #used │ top experts (share)         │
    """
    rows = []
    hot_slots = []
    for il in sorted(layers):
        c = layers[il]
        tot = sum(c.values())
        top = sorted(c.items(), key=lambda kv: -kv[1])[:n_top]
        cell = "  ".join(f"E{e:<3} {cnt / tot * 100:4.1f}%" for e, cnt in top)
        rows.append((str(il), str(len(c)), cell))
        for e, cnt in top:
            hot_slots.append((il, e, cnt, cnt / tot))

    w0 = max(5, *(len(r[0]) for r in rows))
    w1 = max(5, *(len(r[1]) for r in rows))
    w2 = max(len(f"top-{n_top} experts (share of layer's routing)"), *(len(r[2]) for r in rows))

    def line(l, m, r):
        return l + "─" * (w0 + 2) + m + "─" * (w1 + 2) + m + "─" * (w2 + 2) + r

    print(f"\n=== hot experts: {tag} ===")
    print(line("┌", "┬", "┐"))
    print(f"│ {'layer':<{w0}} │ {'#used':<{w1}} │ {f'top-{n_top} experts (share of layer)':<{w2}} │")
    print(line("├", "┼", "┤"))
    for r in rows:
        print(f"│ {r[0]:<{w0}} │ {r[1]:<{w1}} │ {r[2]:<{w2}} │")
    print(line("└", "┴", "┘"))

    print(f"\n  hottest (layer, expert) slots:")
    for il, e, cnt, share in sorted(hot_slots, key=lambda x: -x[2])[:10]:
        print(f"    layer {il:>2}  E{e:<3} : {cnt:>9,} selections  ({share * 100:4.1f}% of layer)")


def cold_report(data, cold_pct):
    """Pruning-oriented report over ALL tags combined: per layer, which experts
    were never selected, and which fall below cold_pct% of the layer's mass."""
    # merge all tags: layer -> expert -> count
    layers = defaultdict(lambda: defaultdict(int))
    for tag in data:
        for il, c in data[tag].items():
            for e, n in c.items():
                layers[il][e] += n
    n_expert = max(max(c) for c in layers.values()) + 1
    total = sum(sum(c.values()) for c in layers.values())

    print(f"\n=== cold experts (all workloads combined: {total:,} selections, n_expert={n_expert}) ===")
    print(f"  cold = < {cold_pct}% of the layer's routed mass  (uniform share would be {100.0/n_expert:.2f}%)")
    print(f"\n  {'layer':>5} | {'never':>5} | {'cold':>4} | never-used expert ids")
    print("  " + "-" * 74)
    tot_never, tot_cold = 0, 0
    prunable = {}
    for il in sorted(layers):
        c = layers[il]
        lt = sum(c.values())
        never = sorted(set(range(n_expert)) - set(c))
        cold  = sorted(e for e, n in c.items() if n / lt * 100.0 < cold_pct)
        tot_never += len(never)
        tot_cold  += len(cold)
        prunable[il] = never + cold
        ids = ",".join(map(str, never)) if never else "-"
        if len(ids) > 44:
            ids = ids[:41] + "..."
        print(f"  {il:>5} | {len(never):>5} | {len(cold):>4} | {ids}")
    print("  " + "-" * 74)
    n_slots = len(layers) * n_expert
    print(f"  totals: never-used {tot_never}/{n_slots} slots, cold {tot_cold} more")
    mean_prun = mean(len(v) for v in prunable.values())
    print(f"  prunable (never+cold) per layer: mean {mean_prun:.1f} of {n_expert} "
          f"({mean_prun/n_expert*100:.0f}% of expert weights)")
    print(f"  NOTE: pruning must be done per layer (slice ffn_*_exps + ffn_gate_inp");
    print(f"        rows with gguf-py); expert ids are per-layer namespaces.")
    return prunable


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
    ap.add_argument("--hot", type=int, nargs="?", const=3, default=None, metavar="N",
                    help="print per-layer top-N expert table (default N=3)")
    ap.add_argument("--cold", type=float, nargs="?", const=0.1, default=None, metavar="PCT",
                    help="pruning report: never-used + experts below PCT%% of layer mass (default 0.1)")
    args = ap.parse_args()

    data = load(args.csvs)
    if not data:
        sys.exit("no rows found")

    top_ks = (8, 16, 32, 64, 96)
    mass_targets = (0.50, 0.80, 0.90, 0.95, 0.99)

    shares_by_tag = {}
    for tag in sorted(data):
        shares_by_tag[tag] = analyze_tag(tag, data[tag], top_ks, mass_targets)

    if args.hot is not None:
        for tag in sorted(data):
            hot_table(tag, data[tag], args.hot)

    if args.cold is not None:
        cold_report(data, args.cold)

    cross_tag(data, args.top_k)

    if args.plot:
        plot(shares_by_tag)


if __name__ == "__main__":
    main()
