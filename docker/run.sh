#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG=${IMAGE_TAG:-baozipu-demo:latest}
CONTAINER_NAME=${CONTAINER_NAME:-baozipu-demo}

GROUP_ARGS=()
declare -A GROUPS_SEEN=()
for device in /dev/dri/card* /dev/dri/render*; do
    [[ -e "${device}" ]] || continue
    gid=$(stat -c '%g' "${device}")
    if [[ -z "${GROUPS_SEEN[${gid}]+x}" ]]; then
        GROUP_ARGS+=(--group-add "${gid}")
        GROUPS_SEEN[${gid}]=1
    fi
done

[[ -d /dev/dri ]] || {
    echo "ERROR: /dev/dri is unavailable; Intel GPU cannot be passed to Docker" >&2
    exit 1
}

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d \
    --name "${CONTAINER_NAME}" \
    --network host \
    --device /dev/dri \
    "${GROUP_ARGS[@]}" \
    --restart unless-stopped \
    "${IMAGE_TAG}"

echo "Container: ${CONTAINER_NAME}"
echo "Web UI:    http://0.0.0.0:7861"
echo "Logs:      docker logs -f ${CONTAINER_NAME}"
