#!/usr/bin/env python3
"""
prune_experts.py — cut cold MoE experts out of a GGUF, guided by a profile.

Reads the usage CSV produced by moe-profiler.h (tag,layer,expert,count),
ranks experts per layer by total selections (all tags merged), removes the
K coldest experts of EVERY layer, and writes a new GGUF with:

  * blk.N.ffn_gate_up_exps.weight  [n_embd, 2*n_ff_exp, E] -> E-K slabs
    (or separate ffn_gate_exps / ffn_up_exps if the model has them)
  * blk.N.ffn_down_exps.weight     [n_ff_exp, n_embd, E]   -> E-K slabs
  * blk.N.ffn_down_exps.scale      [E]                     -> E-K entries
  * blk.N.ffn_gate_inp.weight      [n_embd, E]             -> E-K router rows
  * <arch>.expert_count            E -> E-K

The expert dim is outermost (ne[-1]) in all of these, so each expert is a
contiguous byte slab — quantized tensors (Q4_K, Q8_0, ...) are sliced
without dequantizing. K must be uniform across layers because expert_count
is a global hparam; WHICH experts are cut differs per layer.

Usage:
  python3 prune_experts.py src.gguf dst.gguf --csv moe_profile.csv [more.csv] \
      --prune 16 [--dry-run] [--report pruned.json]

Safety: aborts if any tensor carries the expert dim but is not in the
handled set. Router (ffn_gate_inp) stays F32 — selection order of kept
experts is preserved exactly.

Quality note: after pruning, tokens that would have routed to a removed
expert fall back to the next-best kept one. Prune conservatively and A/B
your workloads before trusting the result.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# gguf-py from the llama.cpp tree this script lives in
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402
from gguf.quants import quant_shape_to_byte_shape  # noqa: E402


def writer_shape(ne, ttype):
    """We always hand GGUFWriter raw uint8 data + raw_dtype; in that mode it
    expects the numpy-order BYTE shape (last dim = bytes per row) and converts
    back to the logical shape internally (quant_shape_from_byte_shape)."""
    return list(quant_shape_to_byte_shape(list(reversed(ne)), ttype))

# tensor-name suffixes that carry the expert dim as ne[-1]
EXPS_3D = ("ffn_gate_up_exps.weight", "ffn_gate_exps.weight",
           "ffn_up_exps.weight", "ffn_down_exps.weight")
EXPS_1D = ("ffn_down_exps.scale", "ffn_gate_up_exps.scale",
           "ffn_gate_exps.scale", "ffn_up_exps.scale", "ffn_exp_probs_b.bias")
ROUTER  = ("ffn_gate_inp.weight",)

SKIP_KEYS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}


def load_counts(paths):
    layers = defaultdict(lambda: defaultdict(int))
    for p in paths:
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                layers[int(row["layer"])][int(row["expert"])] += int(row["count"])
    return layers


def keep_lists(layers, n_layer, n_expert, k):
    """per-layer sorted list of expert ids to KEEP (coldest k removed)"""
    keep, pruned = {}, {}
    for il in range(n_layer):
        c = layers.get(il, {})
        ranked = sorted(range(n_expert), key=lambda e: (c.get(e, 0), e))
        cut = set(ranked[:k])
        keep[il]   = [e for e in range(n_expert) if e not in cut]
        pruned[il] = sorted(cut)
    return keep, pruned


def layer_of(name):
    # blk.<il>.rest
    parts = name.split(".")
    return int(parts[1]) if parts[0] == "blk" and parts[1].isdigit() else None


def slice_expert_dim(t, keep):
    """slice a tensor whose ne[-1] is the expert dim; returns (bytes, new_ne)"""
    ne = [int(d) for d in t.shape]           # ggml ne order, expert dim last
    n_expert = ne[-1]
    raw = t.data.view(np.uint8).reshape(-1)  # no-copy byte view of the memmap
    assert raw.nbytes == t.n_bytes, f"{t.name}: byte count mismatch"
    slab = raw.nbytes // n_expert
    assert raw.nbytes % n_expert == 0, f"{t.name}: not divisible into expert slabs"
    out = np.concatenate([raw[e * slab:(e + 1) * slab] for e in keep])
    return out, ne[:-1] + [len(keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src");  ap.add_argument("dst")
    ap.add_argument("--csv", nargs="+", required=True, help="moe_profile.csv file(s)")
    ap.add_argument("--prune", type=int, required=True, help="experts to remove per layer")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default=None, help="write pruned ids per layer as JSON")
    args = ap.parse_args()

    reader = GGUFReader(args.src)
    arch = reader.fields["general.architecture"].contents()
    n_expert = int(reader.fields[f"{arch}.expert_count"].contents())
    n_layer  = int(reader.fields[f"{arch}.block_count"].contents())
    n_used   = int(reader.fields[f"{arch}.expert_used_count"].contents())
    new_count = n_expert - args.prune
    if new_count < n_used:
        sys.exit(f"cannot prune to {new_count} experts: model routes top-{n_used}")

    layers = load_counts(args.csv)
    total = sum(sum(c.values()) for c in layers.values())
    keep, pruned = keep_lists(layers, n_layer, n_expert, args.prune)

    print(f"arch={arch}  layers={n_layer}  experts {n_expert} -> {new_count}  "
          f"(profile: {total:,} selections)")
    zero_cut = sum(1 for il in pruned for e in pruned[il]
                   if layers.get(il, {}).get(e, 0) == 0)
    print(f"pruning {args.prune}/layer = {args.prune * n_layer} expert slots "
          f"({zero_cut} were never selected in the profile)")

    # safety scan: every tensor with the expert dim must be handled
    handled = EXPS_3D + EXPS_1D + ROUTER
    for t in reader.tensors:
        if int(t.shape[-1]) == n_expert and layer_of(t.name) is not None:
            if not t.name.endswith(handled):
                sys.exit(f"unhandled expert-dim tensor: {t.name} shape={list(t.shape)}")

    if args.dry_run:
        for il in range(min(n_layer, 5)):
            print(f"  layer {il}: prune {pruned[il]}")
        print("  ... (dry run, nothing written)")
        return

    writer = GGUFWriter(args.dst, arch)
    for field in reader.fields.values():
        if field.name in SKIP_KEYS:
            continue
        val = field.contents()
        if field.name == f"{arch}.expert_count":
            val = new_count
        sub = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, val, field.types[0], sub_type=sub)

    n_sliced = 0
    for t in reader.tensors:
        il = layer_of(t.name)
        if il is not None and t.name.endswith(handled) and int(t.shape[-1]) == n_expert:
            data, new_ne = slice_expert_dim(t, keep[il])
            writer.add_tensor(t.name, data, raw_shape=writer_shape(new_ne, t.tensor_type),
                              raw_dtype=t.tensor_type)
            n_sliced += 1
        else:
            writer.add_tensor(t.name, t.data.view(np.uint8).reshape(-1),
                              raw_shape=writer_shape([int(d) for d in t.shape], t.tensor_type),
                              raw_dtype=t.tensor_type)

    print(f"sliced {n_sliced} tensors ({n_sliced // n_layer} per layer); writing {args.dst} ...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"arch": arch, "expert_count": new_count,
                       "pruned_per_layer": {str(k): v for k, v in pruned.items()}}, f, indent=1)
        print(f"pruned-id report -> {args.report}")

    src_sz = Path(args.src).stat().st_size / 2**30
    dst_sz = Path(args.dst).stat().st_size / 2**30
    print(f"done: {src_sz:.2f} GiB -> {dst_sz:.2f} GiB  (-{(1 - dst_sz / src_sz) * 100:.1f}%)")


if __name__ == "__main__":
    main()
