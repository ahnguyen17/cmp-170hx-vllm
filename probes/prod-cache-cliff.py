#!/usr/bin/env python3
"""Prod prefix-cache cliff probe v2. Non-streaming max_tokens=1 request latency ~= prefill time.
Distinct corpus offsets per length -> turn1 guaranteed cold; turn2/3 same doc -> cache test."""
import json, os, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8012"
KEY = sys.argv[2] if len(sys.argv) > 2 else "sophia"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen3.8-27b"
LENS = [(6000, 100000), (20000, 400000), (40000, 800000)]
if len(sys.argv) > 4 and sys.argv[4] == "80k":
    LENS.append((80000, 1200000))

def metrics():
    hits = queries = None
    try:
        req = urllib.request.Request(BASE + "/metrics", headers={"Authorization": f"Bearer {KEY}"})
        with urllib.request.urlopen(req, timeout=8) as r:
            for ln in r.read().decode().splitlines():
                if ln.startswith("vllm:prefix_cache_hits") and "quantile" not in ln:
                    hits = float(ln.rsplit(" ", 1)[1])
                elif ln.startswith("vllm:prefix_cache_queries") and "quantile" not in ln:
                    queries = float(ln.rsplit(" ", 1)[1])
    except Exception as e:
        return f"metrics-unavailable: {str(e)[:60]}"
    return f"hits={hits} queries={queries}"

def latency(prompt):
    payload = json.dumps({
        "model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
        "model": MODEL, "max_tokens": 1, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        json.loads(r.read())
    return time.time() - t0

corpus = open(os.path.expanduser("~/bench/labd_corpus_xl.txt")).read()
for target_tok, off in LENS:
    chars = int(target_tok * 4.61)
    doc = corpus[off:off + chars] + "\n\nReply with just: ok"
    m0 = metrics()
    t1 = latency(doc)
    t2 = latency(doc)
    t3 = latency(doc)
    m1 = metrics()
    speed = target_tok / t1 / 1000
    print(f"[{target_tok:>6} tok @off{off}] cold={t1:7.2f}s ({speed:.1f}k tok/s)  "
          f"turn2={t2:6.2f}s  turn3={t3:6.2f}s  | {m0} -> {m1}")
    sys.stdout.flush()
