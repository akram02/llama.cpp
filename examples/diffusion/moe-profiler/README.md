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
- `analyze_experts.py` — coverage curves, cross-workload Jaccard, verdict

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

## Validated (DiffusionGemma-26B-A4B Q4_K_M)
selections/event = 8 x n_tokens exactly (top-8 routing), events = 30 layers x
denoise steps, expert ids 0..127 — all consistent with model metadata.
The router (`ffn_gate_inp`) is never quantized, so Q4_K_M stats are honest.

## Caveats
- Expert IDs are per-layer namespaces; never aggregate across layers by raw id.
- The eval callback forces per-node syncs: do not benchmark tok/s while profiling.
- Per-expert mixed precision is not possible in stock GGUF (experts live in
  fused per-layer tensors); what IS possible: `--n-cpu-moe`/`-ot` placement,
  and real expert pruning by rewriting the GGUF with gguf-py.
