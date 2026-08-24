# CMP 170HX (Hynix) — vLLM serving recipe

Known-good, production-validated configuration for serving **Qwen3.8-27B-Int8 (GPTQ) with
MTP speculative decoding (k=3) and prefix caching** on a single **NVIDIA CMP 170HX 64GB —
Hynix memory variant**, 24/7.

The 170HX is a GA100 die with 64GB HBM2e that sells for a fraction of an A100. The
**Hynix-memory variant** is the one you want for inference: it runs GPTQ int8 weights with
bf16 activations, CUDA graphs, and fp8 KV cache stably. (Samsung-memory cards we tested
required kernel workarounds — exllama instead of Marlin, eager mode — and still threw
Xid 79 falls-off-the-bus under sustained load. This recipe is validated on **Hynix only**.)

**Current recipe = upstream nightly** (`serve-qwen38-nightly.sh`): the first build where
prefix caching and MTP speculative decode coexist. The 0.27.1 recipe
(`serve-qwen38-mtp.sh`) is retained as the conservative fallback.

**2026-08-23 addendum** ([docs/gdn-prefix-caching-2026-08-23.md](docs/gdn-prefix-caching-2026-08-23.md)):
root-caused the long-context misery — turn-2 prefix-cache miss is upstream
[vllm#45238](https://github.com/vllm-project/vllm/issues/45238) (align-mode single-checkpoint
retention; a prefix needs ~3 sightings before it hits). Added `--prefix-match-unit 400`
(cached TTFT halved; unit must divide the 1600-token GDN block). Confirmed decode is
69–79 tok/s — counting SSE deltas reads a false ~30 because MTP emits ~3.5 tok/delta.
Do NOT raise `--max-num-batched-tokens` above 8192: GDN prefill intermediates scale with
scheduled tokens — 32000 OOM'd in `chunk_fwd_o` and wedged the GA100 card (reboot-only).
Probe scripts in `probes/`.

## Measured performance (single 170HX, Hynix, 64GB)

Same-day A/B, same harness (300-token single-stream gens, 3-run medians):

| Metric | 0.27.1 fork | Nightly (current) |
|---|---|---|
| Decode, prose greedy | 60.0 tok/s | **65.4 tok/s** |
| Decode, prose temp 0.7 | 62.5 tok/s | **66.8 tok/s** |
| Decode, structured greedy | 74.1 tok/s | 74.0 tok/s |
| MTP acceptance | 3.12 / 4 | 3.02 / 4 |
| TTFT, 7K shared prefix, cold | ~4.0 s | ~6.7 s |
| TTFT, 7K shared prefix, **cached** | ~4.0 s (no cache) | **1.44 s** (33.5% hit rate) |
| Context length | 262,144 (fp8 KV) | 262,144 (fp8_e4m3 KV) |

The headline: on workloads that replay a long system prompt every call (agents, RAG,
note-generation pipelines), cached TTFT drops **4.0 s → ~1.4 s** while decode *gains*
7–9% — the old "MTP vs prefix cache, pick one" tradeoff is gone.

## Stack this was validated on

| Component | Version |
|---|---|
| GPU | NVIDIA CMP 170HX 64GB (Hynix) |
| Driver | 610.43.03 |
| vLLM | **0.26.1rc1.dev1024+g1fe3a1571** (upstream nightly wheel) |
| torch | 2.13.0+cu130 |
| Model | Qwen3.8-27B-Int8 (GPTQ, with MTP draft head) |

## Files

- `serve-qwen38-nightly.sh` — **current production launch script** (sanitized: paths/API
  key via env vars)
- `serve-qwen38-mtp.sh` — legacy 0.27.1 launch script (no prefix caching under spec
  decode; battle-tested fallback)
- `patches/vllm-pr50021-gdn-spec-bounds.patch` — required **only for 0.27.1** (GDN-relevant
  subset of [vllm-project/vllm#50021](https://github.com/vllm-project/vllm/pull/50021))
- `patches/vllm-pr50021-full-pr.diff` — complete upstream PR diff for reference
- `serve-g3-int8df2.sh` — **Samsung 40GB 170HX recipe**: int8 + DFlash2 drafter via the
  0.27.1 fork (`SPEC=dflash2 CTX=fast`, port 18020)
- `patches/sm80-int8-repack-cpu-fallback.patch` — **required for the DFlash2 recipe on
  sm80/170HX**: CPU-side Marlin repack (bit-exact) to stop the Xid-31 wedge; also routes
  W8A16 int8 target loads through the same safe path

**Patch status on nightly:** NOT required — the nightly build's GDN rework ran hours of
MTP spec decode (including k=7 stress) without the patch. Keep the patch around for 0.27.1
installs; it becomes unnecessary on any release containing PR #50021 or the rework.

## Quick start

```bash
# 1. venv with the validated nightly build (wheel index serves commit-pinned builds;
#    pin exactly 0.26.1rc1.dev1024+g1fe3a1571 — nightly is a moving target)
uv venv vllm_nightly
uv pip install --python vllm_nightly --extra-index-url https://wheels.vllm.ai/nightly \
  "vllm==0.26.1rc1.dev1024"

# 2. Launch (pin the card by UUID if you have a mixed rig: nvidia-smi -L)
CUDA_VISIBLE_DEVICES=GPU-<uuid> MODEL_PATH=/path/to/Qwen3.8-27B-Int8 API_KEY=secret \
  PATH="$PWD/vllm_nightly/bin:$PATH" bash serve-qwen38-nightly.sh
```

No patch step needed on the nightly build.

## Flag-by-flag: why each non-obvious choice

| Flag | Why |
|---|---|
| `--quantization gptq --dtype bfloat16` | Int8 GPTQ weights with bf16 activations — stable on Hynix HBM2e; Marlin kernels fine here (they wedge Samsung bins) |
| `--max-model-len 262144` | Full 256K context; fits because of the two cache dtypes below |
| `--kv-cache-dtype fp8_e4m3` | Halves KV footprint vs fp16 — what makes 262K context fit in 64GB with headroom |
| `--mamba-ssm-cache-dtype auto` | Nightly resolves GDN/SSM state dtype; fp16 chosen automatically for stability |
| `--enable-prefix-caching` | The point of the nightly migration. Auto-enables `mamba-cache-mode align`, which makes GDN state blocks cacheable alongside KV blocks |
| `--async-scheduling` + `--max-num-batched-tokens 8192` + `--max-num-scheduled-tokens 8192` | **REQUIRED with prefix caching.** With the default chunked-prefill batch, hit rate sits at 0.0% while everything looks healthy: mamba-align block registration needs prefill steps that end exactly on the 1600-token hash boundary. 8192 ≥ typical prompts ⇒ single-shot prefills land on the boundary |
| `--speculative-config mtp k=3 probabilistic` | ~2x decode vs dense. k>3 is a measured dead end with the stock MTP head: acceptance saturates ≈3.2/8 at k=7; k=4 and k=7 both net −27..45% tok/s |
| `max_cudagraph_capture_size: 128` | Graph capture up to batch 128 — production serves concurrent sessions; 32 causes re-capture stalls under burst load |
| `custom_ops +rms_norm +silu_and_mul` | Hand-tuned fused ops for GA100; measurable decode gain over inductor defaults |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Prevents fragmentation OOM over long uptimes on HBM2e |
| `CUDA_DEVICE_ORDER=PCI_BUS_ID` + UUID pin | In mixed rigs (e.g. 170HX beside GeForce cards) FASTEST_FIRST ordering will hand your server the wrong GPU |

## int8 + DFlash2 on the 40GB Samsung (second 170HX)

The Samsung-memory 170HX (40GB) that can't run the Hynix recipe still serves — with the
DFlash2 block-drafting spec-decode head from the [qwen38-27b-rtx3090 fork](https://github.com/syv-ai/qwen38-27b-rtx3090)
and an sm80-specific workaround. Validated Aug 23 2026, 512-token single-stream gens:

| Metric | int8 + DFlash2 (this recipe) | int8 + MTP k=3 (same card) |
|---|---|---|
| Decode, single-stream | **97.8 tok/s** | 53.9 tok/s |
| Aggregate, 8 concurrent streams | **275 tok/s** | — |
| Draft acceptance | 2.9 – 3.1 | 2.7 – 3.0 |
| Context | 64K (bf16 KV, FlashAttention) | 16K |
| Boot time | ~7 min (CPU repack, one-time) | ~4 min |
| Xid 31 wedges | 0 since fix (was: every boot) | — |

### The sm80 Xid-31 bug and the workaround

Loading W4A16/W8A16 Marlin weights on GA100 (sm80) wedges the card: Xid 31
`FAULT_INFO_TYPE_REGION_VIOLATION … VIRT_WRITE` originating in `gptq_marlin_repack`
traffic. The fault is asynchronous — the Python traceback points at a bystander frame
(an `empty_cache`, a `.to()` cast), and the faulting VA drifts with allocation layout,
so no single kernel ever looks guilty. Evidence chain: a bit-exact torch
reimplementation of the repack runs clean on sm86 under `compute-sanitizer` (0 errors),
and clean on sm80 at low occupancy — it faults only under production occupancy
(~27GB resident). Root cause: the repack's GB-scale int64 intermediates churn VMM page
mappings on sm80 until a mapping dies; the next kernel write faults. Fix: run the repack
on CPU (pure integer layout math, bit-exact vs the compiled kernel on real weights) and
copy the result back — one H2D per layer, ~3 extra minutes at boot, zero Xids since.

Two things are mandatory on this card, and one is the opposite of the Hynix recipe:

1. `patches/sm80-int8-repack-cpu-fallback.patch` — re-apply after every fork-venv rebuild:
   `patch -d <venv>/lib/python3.12/site-packages -p1 < patches/sm80-int8-repack-cpu-fallback.patch`
2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` — an *independent* Xid-31 trigger
   on this bin (Hynix wants `:True` for fragmentation; Samsung sm80 wedges with it).

`serve-g3-int8df2.sh` wraps the fork's `single-user/start_qwen.sh`
(`SPEC=dflash2 CTX=fast GPU_UTIL=0.96`, port 18020).

## Troubleshooting

- **Prefix cache hit rate pinned at 0.0% despite `--enable-prefix-caching`** → you dropped
  the `--async-scheduling --max-num-batched-tokens 8192 --max-num-scheduled-tokens 8192`
  trio. Logs will still say caching is enabled; only the hit-rate metric tells the truth.
- **`architecture failed to be inspected` at startup** → your working directory is inside
  the vllm source tree (`site-packages/vllm`) and shadows the `tokenizers` package for the
  registry-inspection subprocess. Launch from any other directory.
- **Server crashes under MTP with errors mentioning `causal_conv1d` / mamba state shapes**
  (0.27.1 only) → the PR #50021 patch is not applied to the vLLM you're actually running.
- **Xid 79 / card falls off the bus** → you are likely on a Samsung-memory 170HX. That
  variant needs kernel/driver workarounds outside this recipe's scope.
- **Xid 31 / illegal memory access during model load on sm80 (170HX)**, traceback landing
  in `gptq_marlin_repack` or a random nearby frame → the fault is asynchronous; **trust
  the Xid, not the stack**. Apply `patches/sm80-int8-repack-cpu-fallback.patch` AND export
  `expandable_segments:False`. See the Samsung recipe section above for the evidence chain.
- **Card wedged after Xid 31 (every CUDA call hangs)** → warm host reboot is the only
  recovery on 170HX. Probe pattern before relaunching: allocate 0.17/1/4 GB, then sync.
- **OOM at load** → confirm `expandable_segments:True` is exported and lower
  `--gpu-memory-utilization` to 0.85.

## License

MIT
