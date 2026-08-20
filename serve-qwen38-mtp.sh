#!/bin/bash
# Qwen3.8-27B-Int8 (GPTQ) + MTP speculative decoding k=3 on a single CMP 170HX 64GB (Hynix).
# Validated in 24/7 production: ~84 tok/s decode, MTP acceptance ~3.0 of 4 draft tokens,
# 262,144-token context, fp8 KV cache. See README for full details.
#
# REQUIREMENTS:
#   - vLLM 0.27.1 (torch 2.13.0+cu130), driver 610.43.03
#   - patches/vllm-pr50021-wheel-only.patch applied to your vLLM install (REQUIRED —
#     without it, unbounded GDN state reads under MTP spec decode will crash the server)
#   - The model directory must contain the MTP draft weights (Qwen3.8-Int8 release with MTP head)
set -e

# --- Card selection (CRITICAL in mixed rigs) -----------------------------------
# If you have more than one GPU, pin the 170HX explicitly by UUID:
#   nvidia-smi -L   →   CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-....
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Allocator setting that prevents fragmentation OOM on the 64GB HBM2e across long uptimes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Config ---------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.8-27B-Int8}"
API_KEY="${API_KEY:?Set API_KEY env var (used for the vLLM OpenAI-compatible endpoint)}"
PORT="${PORT:-8012}"

exec vllm serve "$MODEL_PATH" \
  --quantization gptq --dtype bfloat16 \
  --api-key "$API_KEY" \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype float16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":128,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port "$PORT"
# Tip: --served-model-name accepts multiple aliases if your clients expect different names,
# e.g. --served-model-name qwen3.8-27b my-other-alias
