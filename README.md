# CMP 170HX (Hynix) — vLLM serving recipe

Known-good, production-validated configuration for serving **Qwen3.8-27B-Int8 (GPTQ) with
MTP speculative decoding (k=3)** on a single **NVIDIA CMP 170HX 64GB — Hynix memory
variant**, 24/7.

The 170HX is a GA100 die with 64GB HBM2e that sells for a fraction of an A100. The
**Hynix-memory variant** is the one you want for inference: it runs GPTQ int8 weights with
bf16 activations, CUDA graphs, and fp8 KV cache stably. (Samsung-memory cards we tested
required kernel workarounds — exllama instead of Marlin, eager mode — and still threw
Xid 79 falls-off-the-bus under sustained load. This recipe is validated on **Hynix only**.)

## Measured performance (single 170HX, Hynix, 64GB)

| Metric | Value |
|---|---|
| Decode throughput (MTP k=3, concurrent agent/chat traffic) | **~84 tok/s** |
| MTP acceptance rate | **~3.0 of 4** draft tokens |
| Context length | 262,144 tokens (fp8 KV cache) |
| TTFT, no system prompt | ~0.3 s |
| TTFT, very long system prompt | **~3.4 s** — see caveat below |

**Caveat — prefix caching is disabled under speculative decode in vLLM 0.27.1.** If your
workload replays a huge system prompt every call, TTFT regresses from ~0.3 s to ~3.4 s.
For agent/chat workloads where decode throughput dominates, MTP is a clear win; for
cache-heavy RAG-style workloads, benchmark with `--speculative-config` removed first.

## Stack this was validated on

| Component | Version |
|---|---|
| GPU | NVIDIA CMP 170HX 64GB (Hynix) |
| Driver | 610.43.03 |
| vLLM | 0.27.1 (wheel install) |
| torch | 2.13.0+cu130 |
| Model | Qwen3.8-27B-Int8 (GPTQ, with MTP draft head) |

## Files

- `serve-qwen38-mtp.sh` — the launch script (sanitized: paths/API key via env vars)
- `patches/vllm-pr50021-gdn-spec-bounds.patch` — **required** patch, exactly what our
  production server runs (verified byte-identical against the live install)
- `patches/vllm-pr50021-full-pr.diff` — complete upstream PR diff for reference

The required patch is the **GDN-relevant subset** of
[vllm-project/vllm#50021](https://github.com/vllm-project/vllm/pull/50021)
("[Bugfix] Bound accepted-token state lookups in GDN/KDA spec decode"). Without it,
MTP speculative decode performs unbounded state reads in the GDN kernels and crashes the
server. The full PR additionally hardens Kimi-K3/KDA and Mamba2 state-selection paths
(`mamba_utils.py`) — not exercised by Qwen3.8, so it's not part of our validated subset.
If the PR merges upstream, this patch becomes unnecessary on releases after that point.

## Quick start

```bash
# 1. venv with the validated version
uv venv vllm_env
uv pip install --python vllm_env vllm==0.27.1

# 2. Apply the REQUIRED patch (unbounded GDN state reads under MTP = crash)
#    https://github.com/vllm-project/vllm/pull/50021
SITE=$(vllm_env/bin/python -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
(cd "$SITE" && patch -p1 --dry-run < /path/to/repo/patches/vllm-pr50021-gdn-spec-bounds.patch)
(cd "$SITE" && patch -p1 < /path/to/repo/patches/vllm-pr50021-gdn-spec-bounds.patch)

# 3. Launch (pin the card by UUID if you have a mixed rig: nvidia-smi -L)
CUDA_VISIBLE_DEVICES=GPU-<uuid> MODEL_PATH=/path/to/Qwen3.8-27B-Int8 API_KEY=secret \
  bash serve-qwen38-mtp.sh
```

## Flag-by-flag: why each non-obvious choice

| Flag | Why |
|---|---|
| `--quantization gptq --dtype bfloat16` | Int8 GPTQ weights with bf16 activations — stable on Hynix HBM2e; Marlin kernels fine here (they wedge Samsung bins) |
| `--max-model-len 262144` | Full 256K context; fits because of the two cache dtypes below |
| `--kv-cache-dtype fp8` | Halves KV footprint vs fp16 — what makes 262K context fit in 64GB with headroom |
| `--mamba-ssm-cache-dtype float16` | Qwen3.8's GDN/hybrid layers: keep SSM state fp16, not fp8 (numerical stability) |
| `--speculative-config mtp k=3 probabilistic` | The headline feature: ~84 tok/s vs ~30-35 dense. k=3 balances acceptance (~3.0/4) vs draft cost |
| `max_cudagraph_capture_size: 128` | Graph capture up to batch 128 — production serves concurrent sessions, not single-user; 32 (vLLM default territory) causes re-capture stalls under burst load |
| `custom_ops +rms_norm +silu_and_mul` | Hand-tuned fused ops for GA100; measurable decode gain over inductor defaults |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Prevents fragmentation OOM over long uptimes on HBM2e |
| `CUDA_DEVICE_ORDER=PCI_BUS_ID` + UUID pin | In mixed rigs (e.g. 170HX beside GeForce cards) FASTEST_FIRST ordering will hand your server the wrong GPU |

## Troubleshooting

- **Server crashes under MTP with errors mentioning `causal_conv1d` / mamba state shapes**
  → the PR #50021 patch is not applied to the vLLM you're actually running. Re-check
  `SITE` and that you patched the same venv that launches.
- **Xid 79 / card falls off the bus** → you are likely on a Samsung-memory 170HX. That
  variant needs kernel/driver workarounds outside this recipe's scope.
- **TTFT feels slow on giant system prompts** → expected: no prefix caching under spec
  decode (0.27.1). Drop `--speculative-config` for cache-heavy workloads.
- **OOM at load** → confirm `expandable_segments:True` is exported and lower
  `--gpu-memory-utilization` to 0.85.

## License

MIT
