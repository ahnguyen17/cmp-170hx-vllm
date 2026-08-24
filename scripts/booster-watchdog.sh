#!/bin/bash
# booster-watchdog: restart the sighting-booster proxy (:8013) if it died.
# Runs from cron every 5 min. Only starts if upstream :8012 is healthy.
if ss -ltn 2>/dev/null | grep -q ':8013 '; then
  exit 0
fi
if ! curl -s -m 3 -o /dev/null http://127.0.0.1:8012/health; then
  exit 0  # upstream down; boot-recovery will handle on reboot
fi
echo "[$(date +%F' '%T)] watchdog: 8013 down, restarting" >> /tmp/booster-proxy.log
setsid nohup /home/f0ol/vllm_nightly/bin/python /home/f0ol/scripts/booster-proxy.py >> /tmp/booster-proxy.log 2>&1 &
