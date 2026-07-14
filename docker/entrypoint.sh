#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    exec "$@"
fi

exec python3 dashcam_event_web.py \
    --host 0.0.0.0 \
    --port "${PORT:-7861}" \
    --vlm-backend "${VLM_BACKEND:-omni}" \
    --vlm-every "${VLM_EVERY:-1}" \
    --sample-fps "${SAMPLE_FPS:-6}" \
    --model-path "${MODEL_PATH}" \
    --video "${VIDEO_PATH}" \
    --device "${DEVICE:-GPU}" \
    --max-slice-nums "${MAX_SLICE_NUMS:-1}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-96}" \
    --disable-ocr
