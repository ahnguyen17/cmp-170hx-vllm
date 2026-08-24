#!/bin/bash
# Boot launcher v6 (Aug 23 2026) — Qwen3.8-27B-Int8 + MTP k=3 on UPSTREAM NIGHTLY, :8012 (Hynix 64G, 262K ctx).
# v6: + booster-proxy :8013 (sighting-booster; pmu400 config on 8012 since Aug 23).
# v5: migrated prod fork 0.27.1 -> nightly (prefix caching + MTP coexist; cached TTFT 4.0->1.1s; skill Aug 20 section).
# ROLLBACK: change line below back to serve-qwen38-mtp-prod.sh (fork venv, no prefix cache, battle-tested).
# RETIRED: 3.6-int8 on Samsung (weights deleted Aug 16) + 3.8-bf16 debug on 8021
#          (bf16 serving BANNED — 2× Xid-79 bus drop under GDN decode; club-3090 skill).
# If the int8 build isn't present yet, launch NOTHING (build in progress / failed).
exec >> /tmp/boot-recovery.log 2>&1
sleep 15  # network/driver settle

eval "$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F', ' '{gsub(/ /,"",$1)} /81:00/{h=$2} END{print "HYNIX="h}')"
echo "[$(date +%H:%M:%S)] boot: Hynix(81:00)=${HYNIX:+present}"

[ -z "$HYNIX" ] && { echo "[$(date +%H:%M:%S)] FATAL: Hynix 64G missing — no serve GPU"; exit 1; }
[ ! -f /home/f0ol/models/Qwen3.8-27B-Int8/config.json ] && { echo "[$(date +%H:%M:%S)] int8 build absent — nothing to serve yet"; exit 0; }

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$HYNIX" setsid nohup /home/f0ol/scripts/serve-qwen38-nightly-prod.sh > /tmp/vllm-38-8012-nightly.log 2>&1 &
echo "[$(date +%H:%M:%S)] int8+MTP nightly :8012 launched on Hynix"

# v6: sighting-booster proxy :8013 -> :8012 (GDN align-cache turn-2 fix; docs/scope-longctx-ttft-2026-08-23.md)
if ! ss -ltn 2>/dev/null | grep -q ':8013 '; then
  setsid nohup /home/f0ol/vllm_nightly/bin/python /home/f0ol/scripts/booster-proxy.py > /tmp/booster-proxy.log 2>&1 &
  echo "[$(date +%H:%M:%S)] booster-proxy :8013 launched"
fi
exit 0
