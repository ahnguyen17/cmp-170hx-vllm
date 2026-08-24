#!/bin/bash
# Qwen3.8-27B-Int8 (GPTQ) + MTP k=3 + PREFIX CACHING on a single CMP 170HX 64GB (Hynix).
# Current production recipe (since Aug 20 2026), on upstream vLLM nightly.
# Validated: decode 65-74 tok/s single-stream medians (+7-9% vs 0.27.1 fork), MTP acceptance
# ~3.0 of 4, 262,144-token context, cached TTFT 1.44s on a 7K shared prefix (was 4.0s flat
# on 0.27.1, which cannot combine prefix caching with spec decode). See README for the A/B.
#
# REQUIREMENTS:
#   - vLLM nightly 0.26.1rc1.dev1024+g1fe3a1571 (torch 2.13.0+cu130), driver 610.43.03
#   - PR #50021 patch NOT required on this build (upstream GDN rework; spec decode ran clean
#     for hours without it). Only 0.27.1 installs need the patch.
#   - The model directory must contain the MTP draft weights (Qwen3.8-Int8 release with MTP head)
#
# THE CRITICAL FLAG TRIO — do not remove:
#   --async-scheduling --max-num-batched-tokens 8192 --max-num-scheduled-tokens 8192
# With the default chunked-prefill batch size, prefix caching silently runs at 0.0% hit rate
# (enabled but never matching): the mamba-align split logic needs single-shot prefills that
# end exactly on the 1600-token hash boundary to register cacheable blocks. Measured both ways.
# ALSO: do NOT raise the batched-tokens budget above 8192 — GDN prefill intermediates scale
# with scheduled tokens; 32000 OOM'd in chunk_fwd_o and wedged the GA100 card (reboot-only).
#
# 2026-08-23: added --prefix-match-unit 400 — halves cached TTFT (partial-hit granularity).
# The unit MUST divide the 1600-token GDN block (512 boot-fails with a clear ValueError).
# Turn-2 cache miss remains: upstream vllm#45238 (align-mode single-checkpoint retention).
# Full data: docs/gdn-prefix-caching-2026-08-23.md
set -e

# --- Card selection (CRITICAL in mixed rigs) -----------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Prevents fragmentation OOM on the 64GB HBM2e across long uptimes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=1

# --- Config ---------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.8-27B-Int8}"
API_KEY="${API_KEY:?Set API_KEY env var (used for the vLLM OpenAI-compatible endpoint)}"
PORT="${PORT:-8012}"

exec vllm serve "$MODEL_PATH" \
  --quantization gptq --dtype bfloat16 \
  --api-key "$API_KEY" \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype auto \
  --enable-prefix-caching --enable-chunked-prefill \
  --prefix-match-unit 400 \
  --async-scheduling --max-num-batched-tokens 8192 --max-num-scheduled-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":128,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port "$PORT"
# Tip: --served-model-name accepts multiple aliases if your clients expect different names,
# e.g. --served-model-name qwen3.8-27b my-other-alias
