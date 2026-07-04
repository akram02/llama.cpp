#!/usr/bin/env python3
"""
ab_quality.py — A/B quality harness: same prompts through two GGUFs, compare.

Runs every prompt in a directory tree (one .txt per prompt, dir name = tag,
same layout as run_profiles.sh) through model A and model B with the same
seed, saves both outputs, and writes a side-by-side report with objective
signals. Judgment of "which answer is better" stays with the human — the
harness makes the comparison cheap, not automatic.

Per-output signals:
  * words, distinct-word ratio (degenerate/looping output scores low)
  * max 8-gram repetition count (high = the model got stuck)
  * for outputs containing a ```python block: does it py_compile?

Usage:
  python3 ab_quality.py --bin ./build/bin/llama-diffusion-cli \
      --model-a original.gguf --model-b pruned.gguf \
      --prompts prompts/ --out ab_results/ [-n 512] [--seed 42] \
      -- -ngl 99 -cmoe --no-mmap -fa on -c 2048

Everything after `--` is passed to the CLI verbatim (both models).
Keep MOE_PROFILE off here: profiling overhead would pollute the timings.
"""

import argparse
import py_compile
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


def run_model(bin_path, model, prompt, n_predict, seed, extra):
    cmd = [bin_path, "-m", model, "-p", prompt, "-n", str(n_predict), "-s", str(seed), *extra]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        return None, f"exit {res.returncode}: {res.stderr[-400:]}"
    # perf summary lines are printed to STDOUT after the generation — grab
    # the throughput from them, then strip them from the returned text
    tp = re.search(r"throughput: ([0-9.]+) tok/s", res.stdout + res.stderr)
    text = "\n".join(l for l in res.stdout.splitlines()
                     if not re.match(r"^(total time|throughput):", l)).strip()
    return text, (tp.group(1) + " tok/s" if tp else "?")


def answer_part(text):
    """strip the reasoning channel if present; keep the user-facing answer"""
    parts = re.split(r"<\|channel\|?>\s*(?:final|answer)?", text)
    return parts[-1].strip() if len(parts) > 1 else text


def metrics(text):
    words = re.findall(r"\S+", text)
    if not words:
        return {"words": 0, "distinct": 0.0, "rep8": 0, "py": "-"}
    grams = Counter(tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7)))
    rep8 = max(grams.values()) if grams else 0

    py = "-"
    m = re.search(r"```python\n(.*?)```", text, re.S)
    if m:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(m.group(1))
            tmp = f.name
        try:
            py_compile.compile(tmp, doraise=True)
            py = "compiles"
        except py_compile.PyCompileError:
            py = "SYNTAX ERROR"
        finally:
            Path(tmp).unlink(missing_ok=True)

    return {"words": len(words),
            "distinct": len(set(words)) / len(words),
            "rep8": rep8, "py": py}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("extra", nargs="*", help="args after -- go to the CLI")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name_a, name_b = Path(args.model_a).stem, Path(args.model_b).stem

    rows = []
    for pf in sorted(Path(args.prompts).glob("*/*.txt")):
        tag, pid = pf.parent.name, pf.stem
        prompt = pf.read_text().strip()
        print(f">>> [{tag}/{pid}]", file=sys.stderr)

        outputs = {}
        for label, model in (("a", args.model_a), ("b", args.model_b)):
            text, speed = run_model(args.bin, model, prompt, args.n, args.seed, args.extra)
            if text is None:
                print(f"    model {label} FAILED: {speed}", file=sys.stderr)
                text = ""
            (out / f"{tag}__{pid}.{label}.txt").write_text(text)
            outputs[label] = (text, speed, metrics(answer_part(text)))

        rows.append((tag, pid, outputs))

    # report
    rpt = out / "report.md"
    with open(rpt, "w") as f:
        f.write(f"# A/B quality report\n\nA = `{name_a}`\nB = `{name_b}`\n"
                f"seed={args.seed}, n={args.n}\n\n")
        f.write("| prompt | model | tok/s | words | distinct | rep8 | python |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for tag, pid, o in rows:
            for label in ("a", "b"):
                _, speed, m = o[label]
                f.write(f"| {tag}/{pid} | {label.upper()} | {speed} | {m['words']} "
                        f"| {m['distinct']:.2f} | {m['rep8']} | {m['py']} |\n")
        f.write("\nSide-by-side outputs: `<tag>__<prompt>.a.txt` / `.b.txt`\n"
                "\nRed flags: rep8 >= 4 (looping), distinct < 0.3 (degenerate), "
                "SYNTAX ERROR on code prompts, or words far below the A twin.\n")
    print(f"\nreport -> {rpt}", file=sys.stderr)

    # console summary
    for tag, pid, o in rows:
        ma, mb = o["a"][2], o["b"][2]
        flag = ""
        if mb["rep8"] >= 4 and mb["rep8"] > ma["rep8"] * 2: flag += " B-LOOPING"
        if mb["py"] == "SYNTAX ERROR" and ma["py"] == "compiles": flag += " B-CODE-BROKEN"
        if ma["words"] and mb["words"] < ma["words"] * 0.4: flag += " B-SHORT"
        print(f"{tag}/{pid:<12} A: {ma['words']:>4}w {ma['py']:<12} "
              f"B: {mb['words']:>4}w {mb['py']:<12}{flag}")


if __name__ == "__main__":
    main()
