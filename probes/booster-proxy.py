#!/usr/bin/env python3
"""
Sighting-booster proxy — fixes long-context turn-2 TTFT on the GDN align-mode cache (#45238).

Sits in front of vLLM (default :8013 -> :8012). After each completed /v1/chat/completions
request, it re-sends the accumulated conversation (incl. the assistant reply) as a
max_tokens=1 "primer" in the background. That lifts the prefix to sighting #2/#3, so the
user's NEXT turn (grown prefix) becomes a partial cache hit instead of a full miss.

Measured: 40K-ctx turn-2 TTFT 22.7s -> 6.6s (see docs/scope-longctx-ttft-2026-08-23.md).

Behavior rules:
  - Only fires when serialized conversation > MIN_CHARS (~4K+ tokens); small chats don't need it.
  - At most ONE booster in flight; latest conversation wins (stale ones skipped).
  - If a real user request arrives mid-booster, the booster is cancelled (real traffic first).
  - X-Booster: off header on a request disables boost logic for that request (A/B testing).
  - Booster goes DIRECT to the upstream, never through itself.

Endpoints: everything proxies transparently; plus GET /booster/stats for live counters.
Run: /home/f0ol/vllm_nightly/bin/python booster-proxy.py   (env: UPSTREAM, PORT, MIN_CHARS)
"""
import asyncio, json, os, time
import aiohttp
from aiohttp import web, ClientSession, ClientTimeout

UPSTREAM   = os.environ.get("UPSTREAM", "http://127.0.0.1:8012")
PORT       = int(os.environ.get("PORT", "8013"))
MIN_CHARS  = int(os.environ.get("MIN_CHARS", "20000"))   # ~4.3K tokens
BOOST_TIMEOUT = 900

stats = {"proxy_requests": 0, "boosts_fired": 0, "boosts_done": 0,
         "boost_errors": 0, "boosts_cancelled": 0, "boost_saved_s": 0.0}
state = {"real_in_flight": 0, "boost_task": None, "pending_conv": None}
session: ClientSession = None

def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

_REQ_ID = [0]
def next_req_id():
    _REQ_ID[0] += 1
    return _REQ_ID[0]

def est_tokens(messages):
    try:
        return len(json.dumps(messages)) // 4
    except Exception:
        return 0

def extract_reply(out):
    """This model puts early output in reasoning (content=None until reasoning finishes)."""
    try:
        m = out["choices"][0]["message"]
        return m.get("content") or m.get("reasoning_content") or m.get("reasoning") or ""
    except Exception:
        return ""

# ---------------------------------------------------------------- booster
async def boost_worker():
    first_conv = state["pending_conv"]
    state["pending_conv"] = None
    while first_conv is not None:
        messages, model, auth, est = first_conv
        stats["boosts_fired"] += 1
        t0 = time.time()
        try:
            async with session.post(
                UPSTREAM + "/v1/chat/completions",
                json={"model": model, "messages": messages,
                      "max_tokens": 1, "temperature": 0},
                headers={"Authorization": auth},
                timeout=ClientTimeout(total=BOOST_TIMEOUT),
            ) as r:
                await r.read()
            dt = time.time() - t0
            stats["boosts_done"] += 1
            log(f"BOOST done  est~{est}tok  {dt:.1f}s  status={r.status}")
        except asyncio.CancelledError:
            stats["boosts_cancelled"] += 1
            log(f"BOOST cancelled (real traffic)  est~{est}tok  after {time.time()-t0:.1f}s")
            return  # stop entirely; next real request will reschedule if needed
        except Exception as e:
            stats["boost_errors"] += 1
            log(f"BOOST error: {e!r}")
        # pick up anything newer that queued while we ran
        first_conv = state["pending_conv"]
        state["pending_conv"] = None
    state["boost_task"] = None

def schedule_boost(messages, model, auth):
    if est_tokens(messages) * 4 < MIN_CHARS:
        return
    state["pending_conv"] = (messages, model, auth, est_tokens(messages))
    if state["boost_task"] is None or state["boost_task"].done():
        state["boost_task"] = asyncio.create_task(boost_worker())

def cancel_boost_if_running():
    t = state["boost_task"]
    if t is not None and not t.done():
        t.cancel()

# ---------------------------------------------------------------- proxy
HOP = {"host", "content-length", "transfer-encoding", "connection"}

def fwd_headers(request):
    return {k: v for k, v in request.headers.items() if k.lower() not in HOP}

async def proxy_generic(request):
    url = UPSTREAM + request.path_qs
    body = await request.read() if request.can_read_body else None
    async with session.request(request.method, url, data=body,
                               headers=fwd_headers(request),
                               timeout=ClientTimeout(total=None)) as r:
        resp = web.StreamResponse(status=r.status,
                                  headers={k: v for k, v in r.headers.items()
                                           if k.lower() not in HOP})
        await resp.prepare(request)
        async for chunk in r.content.iter_any():
            await resp.write(chunk)
        await resp.write_eof()
        return resp

async def chat_completions(request):
    rid = next_req_id()
    boosted = request.headers.get("X-Booster", "on").lower() != "off"
    body_raw = await request.read()
    try:
        body = json.loads(body_raw)
    except Exception:
        body = None
    is_stream = bool(body and body.get("stream"))
    model = (body or {}).get("model", "qwen3.8-27b")
    auth = request.headers.get("Authorization", "")

    url = UPSTREAM + request.path_qs
    stats["proxy_requests"] += 1
    state["real_in_flight"] += 1
    cancel_boost_if_running()          # real traffic preempts the primer
    t0 = time.time()
    log(f"[{rid}] enter stream={is_stream} boosted={boosted} bytes={len(body_raw)}")
    try:
        async with session.post(url, data=body_raw, headers=fwd_headers(request),
                                timeout=ClientTimeout(total=None)) as r:
            log(f"[{rid}] upstream status={r.status} after {time.time()-t0:.1f}s")
            resp = web.StreamResponse(status=r.status,
                                      headers={k: v for k, v in r.headers.items()
                                               if k.lower() not in HOP})
            await resp.prepare(request)
            assistant_text = []
            if is_stream and r.status == 200 and boosted and body:
                # forward verbatim; tee-parse deltas to reconstruct the assistant reply
                buf = b""
                async for chunk in r.content.iter_any():
                    await resp.write(chunk)
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            continue
                        try:
                            d = json.loads(payload)
                            delta = d.get("choices", [{}])[0].get("delta", {})
                            txt = delta.get("content") or ""
                            if txt:
                                assistant_text.append(txt)
                        except Exception:
                            pass
                await resp.write_eof()
                log(f"[{rid}] stream done {time.time()-t0:.1f}s reply={len(''.join(assistant_text))}ch")
                if body and body.get("messages"):
                    # boost even without captured reply (reasoning-only streams):
                    # pre-reply prefix is the bulk of the next turn's shared prefix.
                    msgs = list(body["messages"])
                    if assistant_text:
                        msgs = msgs + [{"role": "assistant",
                                        "content": "".join(assistant_text)}]
                    schedule_boost(msgs, model, auth)
                return resp
            else:
                data = await r.read()
                await resp.write(data)
                await resp.write_eof()
                log(f"[{rid}] done {time.time()-t0:.1f}s bytes={len(data)}")
                if (boosted and body and r.status == 200 and not is_stream
                        and body.get("messages")):
                    # boost even without reply text: the pre-reply prefix is the bulk
                    # of the next turn's shared prefix.
                    reply = ""
                    try:
                        reply = extract_reply(json.loads(data))
                    except Exception:
                        pass
                    msgs = list(body["messages"])
                    if reply:
                        msgs = msgs + [{"role": "assistant", "content": reply}]
                    schedule_boost(msgs, model, auth)
                return resp
    except Exception as e:
        log(f"[{rid}] ERROR after {time.time()-t0:.1f}s: {e!r}")
        raise
    finally:
        state["real_in_flight"] -= 1

async def booster_stats(request):
    return web.json_response({**stats,
                              "real_in_flight": state["real_in_flight"],
                              "boost_running": state["boost_task"] is not None
                                               and not state["boost_task"].done()})

async def health(request):
    async with session.get(UPSTREAM + "/health",
                           timeout=ClientTimeout(total=5)) as r:
        return web.Response(status=r.status, text=await r.text())

app = web.Application(client_max_size=256 * 1024 * 1024)
app.router.add_post("/v1/chat/completions", chat_completions)
app.router.add_get("/booster/stats", booster_stats)
app.router.add_get("/health", health)
app.router.add_route("*", "/{tail:.*}", proxy_generic)

async def on_start(app):
    global session
    # force_close: one fresh connection per request — at single-user scale the pool
    # buys nothing and a stale keep-alive conn cost us a hung request once (2026-08-24).
    session = ClientSession(connector=aiohttp.TCPConnector(force_close=True))

app.on_startup.append(on_start)

if __name__ == "__main__":
    log(f"booster-proxy listening on :{PORT} -> {UPSTREAM}  (min_chars={MIN_CHARS})")
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)
