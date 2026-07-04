#pragma once
// ---------------------------------------------------------------------------
// moe-profiler.h — header-only MoE expert-usage profiler for llama.cpp
//
// Counts, per layer, how many times each expert is selected by the router.
// Works by intercepting the "ffn_moe_topk-<layer>" tensors that
// llm_graph_context::build_moe_ffn() emits (I32, [n_expert_used, n_tokens])
// via the backend-scheduler eval callback, so it works on CPU, CUDA, Vulkan,
// Metal — wherever the graph runs — with no model changes.
//
// Usage (after calling moe_profiler_init(ctx_params) before context creation):
//
//   MOE_PROFILE=1 MOE_PROFILE_TAG=bengali ./llama-diffusion-cli -m model.gguf ...
//
// Environment variables:
//   MOE_PROFILE=1              enable (off by default: zero overhead)
//   MOE_PROFILE_OUT=path.csv   output CSV, appended  (default: moe_profile.csv)
//   MOE_PROFILE_TAG=coding     workload tag column   (default: "default")
//   MOE_PROFILE_MIN_NTOK=N     only count graph evals with >= N tokens
//                              (0 = count everything, incl. encoder/prefill)
//
// CSV schema: tag,layer,expert,count   (appended across runs; one run = one tag)
// ---------------------------------------------------------------------------

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct moe_profiler {
    std::mutex mtx;
    std::map<int, std::vector<uint64_t>> counts;  // layer -> per-expert selection count
    uint64_t total_sel = 0;                       // total (token, expert-slot) selections
    uint64_t n_events  = 0;                       // number of topk tensors observed
    std::string out_path = "moe_profile.csv";
    std::string tag      = "default";
    int64_t min_ntok     = 0;

    static moe_profiler & instance() {
        static moe_profiler p;
        return p;
    }

    void add(int layer, const int32_t * idx, int64_t n) {
        std::lock_guard<std::mutex> lock(mtx);
        auto & v = counts[layer];
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = idx[i];
            if (e < 0) {
                continue;
            }
            if ((size_t) e >= v.size()) {
                v.resize((size_t) e + 1, 0);
            }
            v[(size_t) e]++;
        }
        total_sel += (uint64_t) n;
        n_events  += 1;
    }

    void dump() {
        std::lock_guard<std::mutex> lock(mtx);
        if (counts.empty()) {
            fprintf(stderr, "moe-profiler: no MoE selections observed (is this a MoE model?)\n");
            return;
        }
        FILE * f = fopen(out_path.c_str(), "a");
        if (!f) {
            fprintf(stderr, "moe-profiler: cannot open '%s' for append\n", out_path.c_str());
            return;
        }
        fseek(f, 0, SEEK_END);
        if (ftell(f) == 0) {
            fprintf(f, "tag,layer,expert,count\n");
        }
        for (const auto & kv : counts) {
            for (size_t e = 0; e < kv.second.size(); ++e) {
                if (kv.second[e]) {
                    fprintf(f, "%s,%d,%zu,%" PRIu64 "\n", tag.c_str(), kv.first, e, kv.second[e]);
                }
            }
        }
        fclose(f);
        fprintf(stderr,
                "moe-profiler: tag=%s layers=%zu selections=%" PRIu64 " events=%" PRIu64 " -> %s\n",
                tag.c_str(), counts.size(), total_sel, n_events, out_path.c_str());
    }
};

// backend-scheduler eval callback: called twice per tensor (ask, then data-ready)
static inline bool moe_profiler_cb(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    // selection tensor emitted by build_moe_ffn(): "ffn_moe_topk-<layer>"
    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;

    static const bool debug = [] { const char * d = getenv("MOE_PROFILE_DEBUG"); return d && *d == '1'; }();
    if (debug && ask && strstr(t->name, "moe")) {
        static std::mutex dbg_mtx;
        static std::map<std::string, int> seen;
        std::lock_guard<std::mutex> lock(dbg_mtx);
        if (seen[t->name]++ == 0) {
            fprintf(stderr, "moe-profiler[dbg]: node '%s' op=%s type=%s ne=[%" PRId64 ",%" PRId64 "] contig=%d\n",
                    t->name, ggml_op_name(t->op), ggml_type_name(t->type), t->ne[0], t->ne[1],
                    ggml_is_contiguous(t));
        }
    }

    if (ask) {
        return is_topk;  // only request data for the tensors we care about
    }
    // NOTE: ffn_moe_topk is a non-contiguous VIEW [n_expert_used, n_tokens]
    // into the full [n_expert, n_tokens] argsort tensor (row stride nb[1]).
    // Read it row-by-row; do NOT require contiguity.
    if (!is_topk || t->type != GGML_TYPE_I32 || t->ne[0] <= 0 || t->ne[1] <= 0) {
        return true;
    }

    auto & P = moe_profiler::instance();

    // ne[0] = n_expert_used, ne[1] = n_tokens in this graph eval
    if (P.min_ntok > 0 && t->ne[1] < P.min_ntok) {
        return true;
    }

    const int     layer  = atoi(t->name + 13);
    const int64_t n_used = t->ne[0];
    const int64_t n_tok  = t->ne[1];
    const int64_t n      = n_used * n_tok;

    static thread_local std::vector<int32_t> buf;
    buf.resize((size_t) n);

    const size_t row_bytes = (size_t) n_used * sizeof(int32_t);
    const bool   host      = !t->buffer || ggml_backend_buffer_is_host(t->buffer);

    for (int64_t i1 = 0; i1 < n_tok; ++i1) {
        const size_t off = (size_t) i1 * t->nb[1];
        if (host) {
            memcpy(buf.data() + i1 * n_used, (const char *) t->data + off, row_bytes);
        } else {
            ggml_backend_tensor_get(t, buf.data() + i1 * n_used, off, row_bytes);
        }
    }

    // MOE_PROFILE_DEBUG=1 self-test: the topk tensor is a view whose parent is
    // the contiguous [n_expert, n_tokens] argsort. Its first n_used entries per
    // row are ground truth — compare against our strided read, element-exact.
    if (debug && t->view_src && t->view_src->type == GGML_TYPE_I32) {
        const struct ggml_tensor * p = t->view_src;
        const int64_t pe = p->ne[0];  // n_expert
        static thread_local std::vector<int32_t> pbuf;
        pbuf.resize((size_t) ggml_nelements(p));
        if (!p->buffer || ggml_backend_buffer_is_host(p->buffer)) {
            memcpy(pbuf.data(), p->data, ggml_nbytes(p));
        } else {
            ggml_backend_tensor_get(p, pbuf.data(), 0, ggml_nbytes(p));
        }
        uint64_t mism = 0;
        for (int64_t i1 = 0; i1 < n_tok; ++i1) {
            for (int64_t i0 = 0; i0 < n_used; ++i0) {
                if (buf[i1 * n_used + i0] != pbuf[i1 * pe + i0]) {
                    mism++;
                }
            }
        }
        static std::mutex st_mtx;
        static std::map<int, uint64_t> checked, bad;
        std::lock_guard<std::mutex> lock(st_mtx);
        checked[layer] += (uint64_t) n;
        bad[layer]     += mism;
        if (mism) {
            fprintf(stderr, "moe-profiler[selftest]: layer %d MISMATCH %" PRIu64 "/%" PRId64 "\n",
                    layer, mism, n);
        } else if (checked[layer] == (uint64_t) n) {  // first event for this layer
            fprintf(stderr, "moe-profiler[selftest]: layer %d strided read == argsort[0..%" PRId64 ") per row (%" PRId64 " values) OK\n",
                    layer, n_used, n);
        }
    }

    P.add(layer, buf.data(), n);
    return true;  // continue running the graph
}

// call this on llama_context_params BEFORE llama_init_from_model()
static inline void moe_profiler_init(struct llama_context_params & cparams) {
    const char * en = getenv("MOE_PROFILE");
    if (!en || strcmp(en, "1") != 0) {
        return;  // disabled: leave cb_eval untouched, zero overhead
    }

    auto & P = moe_profiler::instance();
    if (const char * v = getenv("MOE_PROFILE_OUT")) {
        P.out_path = v;
    }
    if (const char * v = getenv("MOE_PROFILE_TAG")) {
        P.tag = v;
    }
    if (const char * v = getenv("MOE_PROFILE_MIN_NTOK")) {
        P.min_ntok = atol(v);
    }

    cparams.cb_eval           = moe_profiler_cb;
    cparams.cb_eval_user_data = nullptr;

    atexit([]() { moe_profiler::instance().dump(); });

    fprintf(stderr, "moe-profiler: enabled (tag=%s, out=%s, min_ntok=%" PRId64 ")\n",
            P.tag.c_str(), P.out_path.c_str(), P.min_ntok);
    fprintf(stderr, "moe-profiler: note: eval callback adds sync overhead; do not benchmark tok/s with it on\n");
}
