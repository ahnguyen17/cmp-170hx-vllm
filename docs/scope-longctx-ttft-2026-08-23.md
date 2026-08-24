# Scope: fixing first-word latency on long-context turns (LMCache vs booster proxy)

**Date:** 2026-08-23 (late) · **Context:** follow-up to `gdn-prefix-caching-2026-08-23.md`

## The problem, precisely

Every real chat turn grows its prefix (old convo + new message). vLLM's align-mode cache needs ~3
sightings of a prefix before it hits (#45238), so a real turn-2 pays a full cache-miss prefill.
Measured tonight on prod (40K ctx, grown +2K turn, non-streaming `max_tokens=1`):

| Scenario | real turn-2 TTFT | note |
|---|---|---|
| A: no booster (today's behavior) | **22.7 s** | full miss; 3rd-sighting control = 2.4 s |
| B: one booster re-send of turn-1 | **6.6 s** | 3.4× faster; only the new tail re-prefills |
| C: two boosters | 10.2 s | no better than B (and noisy) |

(Cold-prefill variance was large tonight: 20–95 s for the same 40K — separate issue, likely
eviction churn / thermal; not blocking this decision.)

**Key insight:** the GPU sits idle between user turns. A ~50-line "sighting-booster" middleware —
after each completed generation, silently re-send the accumulated conversation as a `max_tokens=1`
primer — converts that idle time into cache warmth, making the *next* real turn a partial hit.

## Option 1: Booster proxy (vLLM-native cache + primer)

- **What:** transparent HTTP proxy on 8013 (or a Hermes-side hook) that forwards requests to 8012
  and, after each streaming response completes, fires the primer for the updated prefix.
- **Cost:** ~1–2 h to build + test in real usage. Zero new dependencies, zero server changes.
- **Wins:** turn-2 at 40K: 22.7 s → 6.6 s (measured). At 80K expect ~170 s → ~40–60 s (untested).
- **Limits:** turn-1 cold cost (~20–90 s @40K) is physics — nothing fixes first sight. Concurrency=1
  user assumed (fine here). Extra GPU-seconds per turn ≈ one extra prefill (idle time anyway).

## Option 2: LMCache (external KV store)

Research results (docs.lmcache.ai, PyPI, GitHub):

- ✅ **Hybrid GDN supported & validated** — Qwen3.5/3.6 series is literally our architecture string
  (`Qwen3_5ForConditionalGeneration`); mamba recurrent state stored as opaque pages;
  `--separate-object-groups` gives GDN layers independent cache objects.
- ✅ **Connector already in our nightly** — `lmcache_mp_connector.py` present in dev1024; only
  needs `pip install lmcache` + `--kv-transfer-config '{"kv_connector":"LMCacheMPConnector",
  "kv_role":"kv_both"}'`. No rebuild.
- ✅ **RAM available:** 104 GB free — CPU tier (`--l1-size-gb 60`) easily holds ~262K-ctx sessions.
- ⚠️ **Requires `--max-num-batched-tokens 3199`** (= 2N−1, N=1600) for every-block snapshots —
  a regression from our 8192 (prefill throughput −2.6×; also changes the OOM envelope, safer).
- ⚠️ **MTP spec-decode + LMCache coexistence: unvalidated** (neither docs nor issues confirm).
- ⚠️ **Known bugs:** disk-tier restore silently persists ~4% of attention pages (community fix
  repo exists); issue #2845 argues external hits are unsafe unless mamba state restores exactly.
  CPU-tier + current release is the supported path — avoid disk tier.
- ❌ CacheBlend / CacheGen compression not available for hybrid (byte-opaque pages).
- **Cost:** prototype on idle 3090 with `Qwen3.5-0.8B` (validated recipe, N=544) ≈ 1 h to de-risk
  the stack; port to 8012 nightly ≈ 2–4 h incl. MTP coexistence testing + cliff-probe A/B.

## Recommendation

**Do Option 1 first.** It captures most of the user-felt win (measured 3.4× on turn-2) at ~5% of
the complexity, with zero risk to a working prod server. Re-evaluate LMCache only if:
(a) booster fails at 80K scale, (b) we want cross-restart cache persistence, or (c) upstream
#26807 stalls and LMCache's first-party GDN path matures past the current bugs.

## Probe

`probes/prod-grown-prefix.py` — scenarios A/B/C above (argv: BASE KEY MODEL).
