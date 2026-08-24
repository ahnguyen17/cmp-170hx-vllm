#!/usr/bin/env python3
"""Comprehensive perf benchmark of prod Qwen3.8-27B-Int8+MTP via booster proxy :8013.
Tokens ALWAYS measured via usage fields (SSE chunk-counting reads ~30 t/s FALSE).
"""
import json, time, threading, urllib.request, random

PROXY = "http://127.0.0.1:8013"
KEY, MODEL = "sophia", "qwen3.8-27b"
CPT = 3.6  # chars per token for this tokenizer/corpus

corpus = open("/home/f0ol/bench/labd_corpus_xl.txt", encoding="utf-8", errors="ignore").read()

def slice_novel(off, nchars, seed):
    words = corpus[off:off + nchars].split()
    random.Random(seed).shuffle(words)
    return " ".join(words)

def post(body, booster="off", timeout=900):
    req = urllib.request.Request(
        PROXY + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json",
                 "X-Booster": booster})
    return urllib.request.urlopen(req, timeout=timeout)

def stream_call(prompt, max_tokens, booster="off"):
    """Returns ttft, total_t, completion_tokens, chunk_times."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0.7}
    t0 = time.time(); ttft = None; chunks = []; usage = None
    with post(body, booster) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            p = line[5:].strip()
            if p == "[DONE]": continue
            d = json.loads(p)
            if d.get("usage") and d["usage"].get("completion_tokens"):
                usage = d["usage"]["completion_tokens"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                txt = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                if txt and ttft is None:
                    ttft = time.time() - t0
                if txt:
                    chunks.append(time.time())
    total = time.time() - t0
    toks = usage or (len(chunks) * 3.5)  # fallback estimate only
    return ttft, total, toks, chunks

def wall_call(prompt, max_tokens=1, booster="off"):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    t0 = time.time()
    with post(body, booster) as r:
        d = json.loads(r.read())
    return time.time() - t0

def stats():
    with urllib.request.urlopen(PROXY + "/booster/stats", timeout=10) as r:
        return json.load(r)

fmt = lambda *a: print(*a, flush=True)

# ---------------- Phase A: single-stream decode ----------------
fmt("== A: single-stream decode (512 tok, stream, usage-counted) ==")
A = []
chunks_rep1 = []
for i in range(3):
    ttft, tot, toks, chunks = stream_call("Write a vivid, detailed technical adventure story about a GPU cluster at night. Keep going, be specific.", 512)
    if i == 0:
        chunks_rep1 = chunks
    tps = toks / (tot - ttft)
    A.append((ttft, tot, toks, tps))
    fmt(f"  rep{i+1}: TTFT {ttft:5.2f}s | total {tot:6.1f}s | {toks:4d} tok | decode {tps:5.1f} tok/s")
# inter-token latency from rep 1 chunk times
if len(chunks_rep1) > 10:
    gaps = sorted(b - a for a, b in zip(chunks_rep1, chunks_rep1[1:]))
    fmt(f"  chunk-arrival gap p50 {gaps[len(gaps)//2]*1000:4.0f} ms | p95 {gaps[int(len(gaps)*.95)]*1000:4.0f} ms (SSE chunks ~3.5 tok each)")

# ---------------- Phase B: concurrency scaling ----------------
fmt("== B: concurrency scaling (short 2K prompts, 256 tok out) ==")
for level in (1, 2, 4, 8):
    res = [None] * level
    def worker(k):
        p = slice_novel(200000 + k * 9000, 7200, f"conc-{level}-{k}") + "\n\nContinue this document with further detail."
        ttft, tot, toks, _ = stream_call(p, 256)
        res[k] = (ttft, tot, toks)
    th = [threading.Thread(target=worker, args=(k,)) for k in range(level)]
    t0 = time.time()
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - t0
    agg = sum(r[2] for r in res) / wall
    med_tps = sorted(r[2] / (r[1] - r[0]) for r in res)[len(res)//2]
    med_ttft = sorted(r[0] for r in res)[len(res)//2]
    fmt(f"  x{level}: wall {wall:5.1f}s | median stream {med_tps:5.1f} tok/s | aggregate {agg:6.1f} tok/s | median TTFT {med_ttft:4.1f}s")

# ---------------- Phase C: prefill ladder (booster off, cold->warm) ----------------
fmt("== C: prefill ladder (cold sighting #1, then repeat to sighting #3 = warm) ==")
for tok_len, off in ((8000, 50000), (20000, 400000), (40000, 600000), (80000, 750000)):
    p = slice_novel(off, int(tok_len * CPT), f"lad-{tok_len}") + "\n\nSummarize the systems described above."
    cold = wall_call(p)
    _ = wall_call(p)                      # sighting 2
    warm = wall_call(p)                   # sighting 3 -> hit
    fmt(f"  {tok_len//1000:>3}K ctx: cold {cold:6.1f}s | warm(hit) {warm:5.2f}s | prefill ~{tok_len/cold:6.0f} tok/s cold")

# ---------------- Phase D: booster path = the prod experience ----------------
fmt("== D: booster ON (prod default) — real follow-up turn TTFT ==")
base = slice_novel(1050000, int(40000 * CPT), "boost-base") + "\n\nList every failure mode mentioned above."
body = {"model": MODEL, "messages": [{"role": "user", "content": base}],
        "max_tokens": 64, "temperature": 0}
t0 = time.time()
with post(body, "on") as r:
    out = json.loads(r.read())
t1 = time.time() - t0
reply = out["choices"][0]["message"].get("content") or out["choices"][0]["message"].get("reasoning") or ""
b0 = stats()["boosts_done"]
while stats()["boosts_done"] < b0 + 1:
    time.sleep(0.5)
grow = slice_novel(1150000, 8000, "boost-grow")
p2 = base + f"\n\n[Assistant] {reply[:200]}\n\n[User] {grow}\n\nGiven all of the above, what is the single most urgent fix? Answer briefly."
ttft, tot, toks, _ = stream_call(p2, 32, booster="on")
fmt(f"  40K session: turn1 {t1:5.1f}s | boost fired+done | TURN-2 TTFT {ttft:4.1f}s (stream)")

# everyday short chat, boosted
chat_a = slice_novel(1250000, 7000, "chat-a")
t0 = time.time()
with post({"model": MODEL, "messages": [{"role": "user", "content": chat_a + "\n\nSummarize in two sentences."}],
           "max_tokens": 64, "temperature": 0}, "on") as r:
    r.read()
ta = time.time() - t0
b0 = stats()["boosts_done"]
_waited = 0
while stats()["boosts_done"] < b0 + 1 and _waited < 30:
    time.sleep(0.5); _waited += 1  # short convos (<4K tok) never boost — by design
ttft2, _, _, _ = stream_call(chat_a + "\n\nNow summarize in one word.", 16, booster="on")
fmt(f"  ~2K chat:   turn1 {ta:5.1f}s | TURN-2 TTFT {ttft2:4.1f}s")

fmt("BENCH-DONE")
