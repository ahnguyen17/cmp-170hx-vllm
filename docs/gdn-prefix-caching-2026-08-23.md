# GDN prefix caching on the nightly: what's real, what's broken, what we changed

**Date:** 2026-08-23 · **Model:** Qwen3.8-27B-Int8 (GPTQ/Marlin) + MTP k=3 · **Stack:** vLLM `0.26.1rc1.dev1024+g1fe3a1571` (upstream nightly) · **GPU:** CMP 170HX 64GB "Hynix" (sm80/GA100), single card

Follow-up to the nightly-migration recipe (`96ef38f`). Everything below is measured on live prod (`:8012`), non-streaming `max_tokens=1` requests for prefill (unfakeable), `usage.completion_tokens` for decode. Probe scripts are in [`probes/`](../probes/).

---

## TL;DR

| Lever | Verdict |
|---|---|
| `--prefix-match-unit 400` | ✅ **Keep.** Cached TTFT roughly halved. Must divide the 1600-token block (512 boot-fails). |
| `--max-num-batched-tokens 32000` | ❌ **Never.** CUDA OOM in the FLA chunk kernel → hard GA100 card wedge, reboot-only. Keep 8192. |
| `--mamba-cache-mode all` | 🚫 Not implemented for GDN (PR #36649 closed unmerged; #26807 open). No flag fix exists yet. |
| Turn-2 cache miss | 🔗 Upstream bug [vllm#45238](https://github.com/vllm-project/vllm/issues/45238) — align-mode checkpoint retention. Marconi mitigation needs ~3 sightings of a prefix before it hits. |
| Decode speed | ✅ 69–79 tok/s flat from 6K→40K ctx. The "28–30 tok/s" reading is a measurement artifact (see §5). |

---

## 1. The turn-2 cache miss is upstream's bug

`align` mamba cache mode (the default, and the only mode GDN supports) retains **one** mamba state checkpoint per request, saved at the last block-aligned boundary. The Marconi mitigation (#37898) only checkpoints after a prefix is *already known common* — empirically that means a prefix must be seen ~3 times before the fast path engages.

Probe: distinct corpus offsets per length (guaranteed-cold first sighting), same doc sent 3×, non-streaming `max_tokens=1`:

| ctx | turn 1 (cold) | turn 2 | turn 3 |
|---|---|---|---|
| 20K | 12.80 s | 12.94 s ❌ miss | 1.67 s ✅ |
| 40K | 34.58 s | **67.83 s ❌ miss (worse than cold)** | 5.56 s ✅ |
| 80K | 161.10 s | 174.99 s ❌ miss | 5.00 s ✅ |

Real conversations never repeat a prefix exactly (every turn appends), so **every real chat turn pays the full cache-miss prefill**. That — not decode — is the user-visible misery. Turn-2 at 40K being *slower* than cold points at insertion/eviction churn in the align-mode partial-hit machinery (`vllm/v1/core/sched/scheduler.py`, block-aligned chunk splitting; partial hits gated on `hash_block_size < block_size`).

Our block size is **1600 tokens** ("Setting attention block size to 1600 tokens to ensure attention page size ≥ mamba page size") — coarser than the 528–2096 range reported on #45238 for smaller models, which makes the single-checkpoint retention even more brittle.

## 2. `--prefix-match-unit 400` — the one real win available today

Finer prefix-cache key granularity enables the partial-hit path (`hash_block_size 400 < block_size 1600`). Same probe, before → after:

| ctx | turn 3 before | turn 3 with pmu400 |
|---|---|---|
| 20K | 1.67 s | **0.70 s** |
| 40K | 5.56 s | **2.79 s** (1.43 s measured in a later clean warm run) |
| 80K | 5.00 s | **4.72 s** |

It does **not** fix the turn-2 miss (that's checkpoint *retention*, not key granularity). Zero regressions observed; in prod since 2026-08-23.

**Constraint:** the unit must divide every KV-cache group block size. With GDN's 1600-token blocks, valid values are divisors of 1600 (1600, 800, 400, 320, 200, 160, 100, 80, …). `--prefix-match-unit 512` fails at boot:

```
ValueError: Invalid prefix_match_unit=512; all KV cache group block sizes must be
divisible by prefix_match_unit. Got group block sizes=[1600, 1600, 1600, 1600].
```

## 3. Cold prefill collapses superlinearly with length

At 8192-token chunks, cold prefill throughput (measured, not /metrics — those counters are frozen garbage on this nightly):

| ctx | cold prefill | throughput |
|---|---|---|
| 6K | 0.35 s | 17.3k tok/s |
| 20K | 12.9 s | 1.5k tok/s |
| 40K | 39.5 s | 1.0k tok/s |
| 80K | 166 s | 0.5k tok/s |

Engine log during the 80K probe shows 10-second windows of `Running: 1 reqs, prompt throughput: 0.0 tok/s` between 2–7.7k tok/s bursts. Attribution: quadratic intra-chunk cost in the Triton GDN prefill kernels (`_causal_conv1d_fwd`, `chunk_fwd_o`, etc.) plus memory churn — see next section.

## 4. Raising the chunk budget OOMs and **wedges the card**

Theory under test: if per-chunk boundary overhead (mamba-align copies: `precopy_mamba_align`, `postprocess_mamba`) dominates, bigger chunks = fewer boundaries = faster. Tried `--max-num-batched-tokens 32000` (20 × 1600, boundary-aligned):

| ctx | 8192 chunks | 32000 chunks |
|---|---|---|
| 6K cold | 0.35 s | 1.87 s (slower) |
| 20K cold | 12.9 s | 16.1 s (slower) |
| 20K turn-2 | 13.1 s | 29.4 s (2× worse) |
| 40K cold | 39.5 s | **CUDA OOM → engine died → card wedged** |

The GDN/FLA prefill path allocates intermediates proportional to the scheduled batch: `chunk_fwd_o` does `torch.empty_like(v)` where `v` spans the whole scheduled window — at 32K tokens/chunk a 376 MiB allocation failed on a 63.5 GB card with the engine otherwise healthy. **Memory ∝ batched tokens** is also the best explanation for §3's superlinear collapse.

**The wedge:** after the hard OOM, *any* CUDA context init on the card fails (`torch.cuda.init()` → `RuntimeError: CUDA driver initialization failed`) while `nvidia-smi` still enumerates it happily at 16 MiB. Same failure family as the Xid-31 sm80 wedges — **warm reboot is the only recovery**. `boot-recovery.sh` (v5) brings prod back automatically; it now launches with pmu400 included.

**Conclusion: keep `--max-num-batched-tokens 8192`.** The G3 rig's conservative 2048 was the right instinct.

## 5. Decode is 69–79 tok/s — counting SSE deltas lies

With MTP k=3 and near-perfect acceptance, the server emits ~3.5 tokens per streaming delta. A probe that counts deltas as tokens reads a phantom ~28–30 tok/s. Ground truth via `usage.completion_tokens` on identical warm prompts (t(257) − t(1), same doc):

| ctx | decode |
|---|---|
| 6K | 256 tok / 3.24 s = **79.0 tok/s** |
| 40K | 256 tok / 3.73 s = **68.7 tok/s** |

Flat with context, matches the 65–75 spec band. **Always measure decode through `usage`, never through delta counts.**

## 6. Open items

- **#45238 upstream fix** (K checkpoints per request / `all` mode): watch #26807; upgrade nightly when it lands.
- Cold-prefill throughput at long ctx is the remaining wound; nothing tunable server-side tonight — it's the Triton GDN prefill kernels.
- LMCache hybrid layer is a candidate to paper over vllm-side misses if upstream stalls.

## Probes

- `probes/prod-cache-cliff.py` — prefill/cache-cliff probe (argv: BASE KEY MODEL [lens like 80k]). Non-streaming `max_tokens=1`, distinct corpus offsets per length.
- `probes/prod-decode-speed.py` — streaming decode probe. ⚠️ Counts deltas → undercounts by the MTP burst factor (~3.5×); use the §5 method for absolute numbers.
