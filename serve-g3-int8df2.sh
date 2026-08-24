#!/bin/bash
# Samsung 40GB 170HX (GA100/sm80) — Qwen3.8-27B-Int8 + DFlash2 drafter on the 0.27.1 fork. Port 18020.
# See README "int8 + DFlash2 on the 40GB Samsung" for the full recipe and evidence chain.
# MANDATORY on this card: patches/sm80-int8-repack-cpu-fallback.patch applied to the fork venv
# AND expandable_segments:False (both independent Xid-31 triggers if skipped).
# Boot cost ~7 min: w4/w8 Marlin repack runs on CPU (sm80 VMM-corruption workaround).
#
# Env:
#   FORK_DIR    qwen38-27b-rtx3090 fork checkout (contains venv/ and single-user/)
#   MODEL_PATH  Qwen3.8-27B-Int8 model directory
#   G3_PCI      PCI bus of the 170HX (default 82:00); UUID-pinned at runtime
set -e
FORK_DIR=${FORK_DIR:?point at the qwen38-27b-rtx3090 fork checkout}
MODEL_PATH=${MODEL_PATH:?point at the Qwen3.8-27B-Int8 model dir}
G3_PCI=${G3_PCI:-82:00}

U=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F', ' -v p="$G3_PCI" '$1 ~ p{print $2}')
[ -z "$U" ] && { echo "FATAL: card at $G3_PCI not present"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
cd "$FORK_DIR"
exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$U" \
  MODEL="$MODEL_PATH" SPEC=dflash2 CTX=fast GPU_UTIL=0.96 \
  EXTRA_ARGS="--dtype bfloat16" bash single-user/start_qwen.sh
