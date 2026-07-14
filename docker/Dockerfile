ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG GPU_DRIVER_VERSION=26.05.37020.3
ARG IGC_VERSION=2.28.4
ARG INSTALL_GPU_DRIVER=1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    INTEL_OPENVINO_DIR= \
    PYTHONPATH= \
    LD_LIBRARY_PATH= \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_PATH=/opt/models/MiniCPM-o-4_5-OV \
    VIDEO_PATH=/opt/baozipu/街景视频.mp4

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        fonts-noto-cjk \
        libgomp1 \
        libsndfile1 \
        ocl-icd-libopencl1 \
        python3 \
        python3-pip \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/intel-gpu
RUN if [[ "${INSTALL_GPU_DRIVER}" == "1" ]]; then \
        wget --tries=3 --timeout=30 -q "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-core-2_${IGC_VERSION}+20760_amd64.deb" \
        && wget --tries=3 --timeout=30 -q "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-opencl-2_${IGC_VERSION}+20760_amd64.deb" \
        && wget --tries=3 --timeout=30 -q "https://github.com/intel/compute-runtime/releases/download/${GPU_DRIVER_VERSION}/intel-ocloc_${GPU_DRIVER_VERSION}-0_amd64.deb" \
        && wget --tries=3 --timeout=30 -q "https://github.com/intel/compute-runtime/releases/download/${GPU_DRIVER_VERSION}/intel-opencl-icd_${GPU_DRIVER_VERSION}-0_amd64.deb" \
        && wget --tries=3 --timeout=30 -q "https://github.com/intel/compute-runtime/releases/download/${GPU_DRIVER_VERSION}/libigdgmm12_22.9.0_amd64.deb" \
        && wget --tries=3 --timeout=30 -q "https://github.com/intel/compute-runtime/releases/download/${GPU_DRIVER_VERSION}/libze-intel-gpu1_${GPU_DRIVER_VERSION}-0_amd64.deb" \
        && apt-get update \
        && apt-get install -y --no-install-recommends ./*.deb; \
    fi \
    && rm -rf /tmp/intel-gpu /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip uninstall --break-system-packages -y opencv-python \
    && python3 -m pip install --break-system-packages --force-reinstall --no-deps \
        opencv-python-headless==4.13.0.92 \
    && rm -f /tmp/requirements.txt

WORKDIR /opt/baozipu
COPY --from=demo dashcam_event_web.py dashcam_event_web.html duplex_baozi_video_demo.py minicpm_o_4_5_helper.py ./
COPY --from=demo 街景视频.mp4 ./街景视频.mp4
COPY --from=model . /opt/models/MiniCPM-o-4_5-OV
COPY entrypoint.sh /usr/local/bin/baozipu-entrypoint

RUN chmod 755 /usr/local/bin/baozipu-entrypoint \
    && test -f "${MODEL_PATH}/openvino_llm_language_model.xml" \
    && test -f "${VIDEO_PATH}"

EXPOSE 7861

ENTRYPOINT ["/usr/local/bin/baozipu-entrypoint"]
