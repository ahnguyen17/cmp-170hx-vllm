#!/usr/bin/env python3
"""A/B verification of the booster proxy: real grown-prefix turn-2, booster off vs on."""
import json, sys, time, urllib.request

PROXY = "http://127.0.0.1:8013"
KEY = "sophia"
corpus = open("/home/f0ol/bench/labd_corpus_xl.txt").read()
CPT = 4.61

def call(doc, max_tokens=1, booster="on", stream=False):
    payload = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": doc}],
                          "max_tokens": max_tokens, "temperature": 0,
                          "ignore_eos": True, "stream": stream}).encode()
    req = urllib.request.Request(PROXY + "/v1/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                 "X-Booster": booster})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        body = r.read()
    dt = time.time() - t0
    reply = ""
    try:
        d = json.loads(body)
        reply = (d["choices"][0]["message"].get("content")
                 or d["choices"][0]["message"].get("reasoning") or "")
    except Exception:
        pass
    return dt, reply

def stats():
    with urllib.request.urlopen(PROXY + "/booster/stats", timeout=10) as r:
        return json.load(r)

import random

def shuffled_slice(off, nchars, seed):
    """Corpus is out of virgin regions — word-shuffled slices are novel token sequences."""
    words = corpus[off:off + nchars].split()
    random.Random(seed).shuffle(words)
    return " ".join(words)

def conv(base_tok, off, tag):
    base = shuffled_slice(off, int(base_tok * CPT), seed=f"base-{tag}")
    p1 = base + f"\n\n[Q1-{tag}] Summarize the key claims above."
    growth = shuffled_slice(off + int(base_tok * CPT) + 500, int(2000 * CPT), seed=f"grw-{tag}")
    p2 = p1 + f"\n\n[Assistant-{tag}] The document covers three systems and their failure modes.\n\n[New] {growth}\n\n[Q2-{tag}] Given all the above, what is the recommended action?"
    return p1, p2

BASE = 20000

print("=== LEG 1: CONTROL (X-Booster: off) — offset 1296K ===")
p1, p2 = conv(BASE, 1296000, "ctl")
t1, reply1 = call(p1, max_tokens=64, booster="off", stream=False)
print(f"  turn1 (32K cold, gen 64tok) = {t1:6.1f}s   reply={len(reply1)} chars")
t2_ctl, _ = call(p2, max_tokens=1, booster="off")
print(f"  turn2 (grown +2K)           = {t2_ctl:6.1f}s   <- CONTROL (expect full miss)")
b0 = stats()["boosts_fired"]

print("=== LEG 2: TEST (booster on) — offset 1400K ===")
p1, p2 = conv(BASE, 1400000, "tst")
t1, reply1 = call(p1, max_tokens=64, booster="on", stream=False)
print(f"  turn1 (32K cold, gen 64tok) = {t1:6.1f}s   reply={len(reply1)} chars (booster fires now)")
# wait for the booster to complete
deadline = time.time() + 600
while time.time() < deadline:
    s = stats()
    if s["boosts_done"] > 0:
        break
    time.sleep(2)
print(f"  booster state: {stats()}")
t2_tst, _ = call(p2, max_tokens=1, booster="on")
print(f"  turn2 (grown +2K)           = {t2_tst:6.1f}s   <- TEST (expect partial hit)")

print(f"\nRESULT: control turn2 = {t2_ctl:.1f}s | boosted turn2 = {t2_tst:.1f}s | speedup = {t2_ctl/t2_tst:.1f}x")
