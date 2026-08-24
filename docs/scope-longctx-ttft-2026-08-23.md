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

---

## UPDATE 2026-08-24: DEPLOYED TO PROD ✅

**Option 1 built, verified, and promoted to the default path** (`55543a3`, `4b9c9a0`).

### End-to-end A/B (final, through the proxy, word-shuffled virgin docs — `probes/booster-ab-test.py`)

| | turn-2 TTFT (20K ctx, grown +2K) |
|---|---|
| X-Booster: off | 12.6 s |
| booster on | **4.0 s (3.1×)** |

Combined with the earlier direct measurements: follow-up turns now land in ~4–7 s instead of
12–23 s, decaying toward the ~1.4–2.7 s warm-hit floor as the session grows.

### Rollout state
- `scripts/booster-proxy.py` — proxy :8013 → :8012; booster fires after each completed chat
  gen (max_tokens=1 primer); cancels on real traffic; one-in-flight; >4K-token gate;
  `X-Booster: off` escape hatch; per-request stage logging; fresh-connection-per-request.
- `scripts/booster-watchdog.sh` — cron \*/5 restarts the proxy if it dies (gated on :8012 health).
- `scripts/boot-recovery.sh` v6 — launches the proxy at boot alongside the engine.
- Hermes `providers.uncensored-27b` + `deepseek-v4-flash` → `http://localhost:8013/v1`;
  verified with the served alias `qwen3.6-27b-uncensored` (PROD-PATH-OK).
- Stats: `curl :8013/booster/stats`.

### Bugs found during verification (fixed, documented for posterity)
1. **Qwen reasoning models reply in `reasoning`** until reasoning finishes (`content: None`)
   — the boost gate originally required `message.content` and never fired at low max_tokens.
   `extract_reply()` now falls back, and the boost fires even with no reply text (the
   pre-reply prefix is the bulk of the next turn's shared prefix).
2. One request hung with no forensics on the pooled-connection path → proxy now logs every
   stage and uses `force_close` connections (stale keep-alive class eliminated).

### Still open
- Upstream #45238 comment (staged) — the 3-sighting align-cache behavior is the root cause;
  this is a workaround, not a fix.
- LMCache (Option 2) parked: revisit at 80K+ sessions or for cross-restart persistence.
- `deepseek-v4-flash` provider model name 404s (stale alias, pre-existing, unrelated).
