#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
MODEL_DIR=${MODEL_DIR:-"${PROJECT_DIR}/../MiniCPM-o-4_5-OV"}
IMAGE_TAG=${IMAGE_TAG:-baozipu-demo:latest}
BASE_IMAGE=${BASE_IMAGE:-ubuntu:24.04}
INSTALL_GPU_DRIVER=${INSTALL_GPU_DRIVER:-1}
PROXY_URL=${PROXY_URL-http://proxy.cd.intel.com:911}

PROXY_ARGS=()
if [[ -n "${PROXY_URL}" ]]; then
    NO_PROXY_VALUE=${NO_PROXY:-${no_proxy:-localhost,127.0.0.1}}
    PROXY_ARGS+=(
        --build-arg "HTTP_PROXY=${PROXY_URL}"
        --build-arg "HTTPS_PROXY=${PROXY_URL}"
        --build-arg "NO_PROXY=${NO_PROXY_VALUE}"
        --build-arg "http_proxy=${PROXY_URL}"
        --build-arg "https_proxy=${PROXY_URL}"
        --build-arg "no_proxy=${NO_PROXY_VALUE}"
    )
fi

[[ -f "${PROJECT_DIR}/dashcam_event_web.py" ]] || {
    echo "ERROR: demo source not found under ${PROJECT_DIR}" >&2
    exit 1
}
[[ -f "${PROJECT_DIR}/街景视频.mp4" ]] || {
    echo "ERROR: demo video not found: ${PROJECT_DIR}/街景视频.mp4" >&2
    exit 1
}
[[ -f "${MODEL_DIR}/openvino_llm_language_model.xml" ]] || {
    echo "ERROR: MiniCPM-o OpenVINO model not found under ${MODEL_DIR}" >&2
    exit 1
}

echo "Building ${IMAGE_TAG}"
echo "  demo:  ${PROJECT_DIR}"
echo "  model: ${MODEL_DIR}"
echo "  base:  ${BASE_IMAGE}"
echo "  install GPU driver: ${INSTALL_GPU_DRIVER}"
echo "  proxy: ${PROXY_URL:-disabled}"

docker build \
    --network=host \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "INSTALL_GPU_DRIVER=${INSTALL_GPU_DRIVER}" \
    "${PROXY_ARGS[@]}" \
    --build-context "demo=${PROJECT_DIR}" \
    --build-context "model=${MODEL_DIR}" \
    --tag "${IMAGE_TAG}" \
    "${SCRIPT_DIR}"

echo
echo "Built ${IMAGE_TAG}"
echo "Run: ${SCRIPT_DIR}/run.sh"
