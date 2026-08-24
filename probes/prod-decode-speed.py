#!/usr/bin/env python3
"""Decode-speed probe: warm-cache TTFT + decode tok/s at increasing ctx (isolates decode)."""
import json, os, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8012"
KEY = sys.argv[2] if len(sys.argv) > 2 else "sophia"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen3.8-27b"

def stream_timings(prompt, max_tokens):
    payload = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time(); ttft = None; ntok = 0; last = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            ln = raw.decode(errors="ignore")
            if ln.startswith("data:") and "[DONE]" not in ln:
                try:
                    d = json.loads(ln[5:])
                    delta = d.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content") or delta.get("reasoning"):
                        ntok += 1
                        if ttft is None:
                            ttft = time.time() - t0
                        last = time.time()
                except Exception:
                    pass
    return ttft, (last - ttft) if ttft and last and last > ttft else 0.001, ntok, time.time() - t0

corpus = open(os.path.expanduser("~/bench/labd_corpus_xl.txt")).read()
# reuse v2-probe offsets -> cache already warm on prod; prefill cheap, decode isolated
for target_tok, off in [(6000, 100000), (20000, 400000), (40000, 800000), (80000, 1200000)]:
    chars = int(target_tok * 4.61)
    doc = corpus[off:off + chars] + "\n\nContinue this document with one more paragraph of technical detail."
    ttft, tdec, ntok, twall = stream_timings(doc, 128)
    print(f"[{target_tok:>6} tok] TTFT={ttft:7.2f}s  decode={ntok}/{tdec:5.2f}s = {ntok/tdec:6.1f} tok/s  wall={twall:7.2f}s")
    sys.stdout.flush()
