#!/usr/bin/env python3
"""Grown-prefix probe: does a REAL conversation turn-2 (prefix grown, not repeated) hit the cache?
Also tests the 'sighting-booster' trick: re-sending the completed prefix once so the next
real turn becomes sighting #3 (the Marconi/#45238 threshold). Non-streaming max_tokens=1.
argv: BASE KEY MODEL
"""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8012"
KEY  = sys.argv[2] if len(sys.argv) > 2 else "sophia"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen3.8-27b"

def latency(doc):
    payload = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": doc}],
                          "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        json.load(r)
    return time.time() - t0

corpus = open("/home/f0ol/bench/labd_corpus_xl.txt").read()
CPT = 4.61  # chars per token

def build(base_tok, off, grow_tok, tag):
    """turn-1 doc, and turn-2 doc = turn-1 + assistant reply + new user material (grown prefix)."""
    p1 = corpus[off:off + int(base_tok * CPT)] + f"\n\n[Q1-{tag}] Summarize the key technical claims above."
    growth = corpus[off + int(base_tok * CPT) + 5000: off + int(base_tok * CPT) + 5000 + int(grow_tok * CPT)]
    p2 = p1 + f"\n\n[Assistant-{tag}] The document covers three systems and their failure modes; details follow.\n\n[New material] {growth}\n\n[Q2-{tag}] Given all the above and the new material, what is the recommended action?"
    return p1, p2

# fresh offsets so nothing is pre-warmed
print("=== Scenario A: baseline conversation (no booster) — 40K turn-1, grown turn-2 ===")
p1, p2 = build(40000, 150000, 2000, "A")
t1  = latency(p1); print(f"  turn1  (40K, cold)        = {t1:7.2f}s")
t2  = latency(p2); print(f"  turn2  (grown +2K, real)  = {t2:7.2f}s   <- the user-felt number today")
t2b = latency(p2); print(f"  turn2 rep (3rd sighting)  = {t2b:7.2f}s   <- control (should hit)")

print("=== Scenario B: sighting-booster (re-send turn-1 once after it completes) ===")
p1, p2 = build(40000, 520000, 2000, "B")
t1  = latency(p1); print(f"  turn1  (40K, cold)        = {t1:7.2f}s")
tb  = latency(p1); print(f"  booster (same 40K again)  = {tb:7.2f}s   (2nd sighting, likely miss; runs while user reads)")
t2  = latency(p2); print(f"  turn2  (grown +2K, real)  = {t2:7.2f}s   <- does the booster make it hit?")

print("=== Scenario C: double booster (turn-1 sent 3x total) ===")
p1, p2 = build(40000, 1000000, 2000, "C")
t1  = latency(p1); print(f"  turn1  (40K, cold)        = {t1:7.2f}s")
tb1 = latency(p1); tb2 = latency(p1)
print(f"  boosters (2nd,3rd sight)  = {tb1:6.2f}s / {tb2:6.2f}s")
t2  = latency(p2); print(f"  turn2  (grown +2K, real)  = {t2:7.2f}s   <- shared prefix now has 3 sightings")
