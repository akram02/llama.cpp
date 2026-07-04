# moe-expert-profiler for llama.cpp (DiffusionGemma / any GGUF MoE)

Counts which experts the router actually selects, per layer, per workload —
directly inside llama.cpp on the GGUF you already have. No PyTorch needed.

Hooks the backend-scheduler eval callback (`cb_eval`) and intercepts the
`ffn_moe_topk-<layer>` selection tensors emitted by `build_moe_ffn()`.
NOTE: that tensor is a NON-CONTIGUOUS VIEW `[n_expert_used, n_tokens]` into
the `[n_expert, n_tokens]` argsort tensor — the profiler reads it row-by-row
(this was the critical fix; a contiguity check would silently count nothing).

## Files
- `../moe-profiler.h` — header-only profiler (wired into diffusion-cli.cpp)
- `run_profiles.sh` — batch runner, one prompt dir per workload tag
- `analyze_experts.py` — coverage/Jaccard/verdict + `--hot` and `--cold` reports
- `prune_experts.py` — profile-guided expert pruning (writes a new GGUF)
- `ab_quality.py` — A/B quality harness (original vs pruned)
- `prompts/` — starter workload prompt sets (bengali/coding/english/math);
  add your own: one `.txt` per prompt, directory name = workload tag
- `example_profile.csv` — real profile from one DiffusionGemma run, so you can
  try `analyze_experts.py example_profile.csv --hot --cold` with no GPU

## Run
```bash
MOE_PROFILE=1 MOE_PROFILE_TAG=coding ./build/bin/llama-diffusion-cli -m model.gguf ...
# batch:
./run_profiles.sh ./build/bin/llama-diffusion-cli model.gguf prompts -ngl 99 -n 256 ...
# analyze:
python3 analyze_experts.py moe_profile.csv
```

Env: `MOE_PROFILE=1` (enable), `MOE_PROFILE_OUT`, `MOE_PROFILE_TAG`,
`MOE_PROFILE_MIN_NTOK` (e.g. 256 = only canvas-sized evals).

## Example: profile one prompt, print the hot-expert table

```bash
# 1. profile a single prompt (writes prime_profile.csv)
MOE_PROFILE=1 MOE_PROFILE_TAG=prime_check MOE_PROFILE_OUT=prime_profile.csv \
./build/bin/llama-diffusion-cli \
    -m diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
    -p "write a prime number check python code" \
    -ngl 99 -sm none -mg 0 -cmoe --no-mmap -fa on -c 2048 -n 256

# 2. per-layer top-expert table (--hot N for top-N, default 3)
python3 analyze_experts.py prime_profile.csv --hot
```

Output (DiffusionGemma-26B-A4B Q4_K_M, single run, 256 output tokens):

```
=== hot experts: prime_check ===
┌───────┬───────┬──────────────────────────────────────────┐
│ layer │ #used │ top-3 experts (share of layer)           │
├───────┼───────┼──────────────────────────────────────────┤
│ 0     │ 101   │ E103 11.1%  E114 11.1%  E47   9.5%       │
│ 1     │ 120   │ E11   9.9%  E120  9.6%  E56   9.6%       │
│ 2     │ 117   │ E76   7.1%  E55   6.6%  E102  5.9%       │
│ 3     │ 117   │ E30   4.6%  E120  4.4%  E13   4.3%       │
│ 4     │ 92    │ E119 10.6%  E23  10.5%  E55  10.4%       │
│ ...   │ ...   │ ...                                      │
│ 29    │ 124   │ E14   7.1%  E29   6.8%  E86   6.2%       │
└───────┴───────┴──────────────────────────────────────────┘

  hottest (layer, expert) slots:
    layer  0  E103 :     4,468 selections  (11.1% of layer)
    layer  0  E114 :     4,458 selections  (11.1% of layer)
    layer  4  E119 :     4,272 selections  (10.6% of layer)
    ...
```

Reading it: `#used` = how many of the 128 experts fired at least once in that
layer; shares are of that layer's total routed selections (top-8 routing,
so a uniform spread would give each expert ~0.78%). Early layers concentrate
hardest (a single expert taking ~11% = ~14x uniform); late layers are flatter.
One short run is NOT a stable hot set — aggregate many prompts per tag before
making placement/pruning decisions.

## Validated (DiffusionGemma-26B-A4B Q4_K_M)
selections/event = 8 x n_tokens exactly (top-8 routing), events = 30 layers x
denoise steps, expert ids 0..127 — all consistent with model metadata.
The router (`ffn_gate_inp`) is never quantized, so Q4_K_M stats are honest.

## Example: find prunable experts (--cold)

`--cold [PCT]` merges ALL tags and reports, per layer, the experts that were
never selected plus those below PCT% of the layer's routed mass (default 0.1):

```bash
python3 analyze_experts.py moe_profile.csv prime_profile.csv --cold
```

Output (DiffusionGemma-26B-A4B, 9.59M selections from 7 runs):

```
=== cold experts (all workloads combined: 9,591,360 selections, n_expert=128) ===
  cold = < 0.1% of the layer's routed mass  (uniform share would be 0.78%)

  layer | never | cold | never-used expert ids
  --------------------------------------------------------------------------
      0 |    15 |   32 | 36,37,40,42,56,57,62,65,80,82,88,93,96,98...
      1 |     7 |   15 | 17,19,43,58,75,95,126
      2 |     3 |   16 | 22,30,94
      3 |     1 |   23 | 95
   ...
  --------------------------------------------------------------------------
  totals: never-used 70/3840 slots, cold 790 more
  prunable (never+cold) per layer: mean 28.7 of 128 (22% of expert weights)
```

"never" shrinks as the profile grows — with little data many experts just
haven't had their moment yet. The cold-share threshold is the robust
criterion; collect a large profile (days/weeks of real usage) before pruning.

## Pruning cold experts (prune_experts.py)

Turns the profile into a smaller GGUF: removes the K coldest experts of every
layer (per-layer cut lists, uniform K — `expert_count` is a global hparam),
slicing the quantized expert tensors byte-slab-wise without dequantizing:

```bash
python3 prune_experts.py src-Q4_K_M.gguf pruned-Q4_K_M.gguf \
    --csv moe_profile.csv --prune 16 --report pruned.json [--dry-run]
```

Slices per layer: `ffn_gate_up_exps.weight`, `ffn_down_exps.weight`,
`ffn_down_exps.scale`, `ffn_gate_inp.weight` (router rows), and updates
`<arch>.expert_count`. Aborts if any unhandled tensor carries the expert dim.

Verified on DiffusionGemma-26B-A4B Q4_K_M (`--prune 16`, profile of ~9.6M
selections): 15.65 -> 13.89 GiB (-11.3%), all offsets consistent, loads and
generates coherent output, expert ids 0..111, and ran ~30% faster with
CPU-offloaded experts. WARNING: tokens that would route to a removed expert
fall back to worse substitutes — prune conservatively and A/B your workloads
before trusting a pruned model. Needs numpy (and tqdm for the progress bar).

## A/B quality testing (ab_quality.py)

Before trusting a pruned model, run the same prompts through both GGUFs and
compare. The harness automates the comparison; the judgment stays with you.

```bash
python3 ab_quality.py --bin ./build/bin/llama-diffusion-cli \
    --model-a original-Q4_K_M.gguf \
    --model-b pruned-Q4_K_M.gguf \
    --prompts prompts/ --out ab_results/ -n 512 --seed 42 \
    -- -ngl 99 -cmoe --no-mmap -fa on -c 2048
```

* Same prompt layout as `run_profiles.sh` (`prompts/<tag>/*.txt`); both models
  get the same seed. Args after `--` go to the CLI verbatim.
* Keep `MOE_PROFILE` OFF here — profiling overhead would pollute timings.

Outputs in `ab_results/`:
* `<tag>__<prompt>.a.txt` / `.b.txt` — full outputs, side by side
* `report.md` — per-run table: tok/s, words, distinct-word ratio, max 8-gram
  repetition, and whether a ```python block compiles (`py_compile`)

Red flags for the B (pruned) model:
* `rep8 >= 4` and much higher than A — the model loops (degeneration)
* `distinct < 0.3` — token soup
* `SYNTAX ERROR` where A compiles — code ability damaged
* words far below A on the same prompt — early collapse

The console summary prints `B-LOOPING` / `B-CODE-BROKEN` / `B-SHORT` flags per
prompt. Objective signals catch collapse, not subtle quality loss — read a few
side-by-side outputs (especially your most important workload) before deciding.

### Verified result (128-expert original vs 112-expert pruned, seed 42, n=512)

| prompt | A words/distinct | B words/distinct | verdict |
|---|---|---|---|
| bengali/01 | 234 / 0.74 | 251 / 0.74 | equivalent |
| bengali/02 | 232 / 0.77 | 216 / 0.79 | equivalent |
| coding/01  | 247 / 0.59 | 240 / 0.63 | equivalent (SYNTAX ERROR on BOTH = n=512 truncation, not pruning) |
| coding/02  | 228 / 0.67 | 231 / 0.72 | equivalent |
| english/01 | 287 / 0.74 | 292 / 0.72 | equivalent |
| math/01    | 235 / 0.62 | 240 / 0.64 | equivalent |

No looping (rep8=1 everywhere), no degeneration, no length collapse — the
16/layer prune shows no measurable damage on these workloads. Caveat: 6 short
prompts is a smoke test, not proof; the pruning profile itself was thin
(~9.6M selections), and truncation-limited runs (use larger -n for code).

## Caveats
- Expert IDs are per-layer namespaces; never aggregate across layers by raw id.
- The eval callback forces per-node syncs: do not benchmark tok/s while profiling.
- Per-expert mixed precision is not possible in stock GGUF (experts live in
  fused per-layer tensors); what IS possible: `--n-cpu-moe`/`-ot` placement,
  and real expert pruning by rewriting the GGUF with gguf-py.
