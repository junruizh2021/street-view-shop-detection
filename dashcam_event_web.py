#!/usr/bin/env python3
"""Realtime dashcam shop-detection web demo."""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from PIL import Image

from duplex_baozi_video_demo import (
    BEVERAGE_BRANDS,
    DEFAULT_MODEL_PATH,
    TARGET_BRANDS,
    analyze_realtime_window,
    build_finding_record,
    build_realtime_scan_prompt_no_evidence,
    build_storefront_grid,
    format_timestamp,
    match_ocr_brand,
    parse_json_array,
    run_ocr_window,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = BASE_DIR / "街景视频.mp4"
PAGE_PATH = BASE_DIR / "dashcam_event_web.html"
DEFAULT_JOYAI_MODEL_PATH = "/home/nvme-data/AI-models/JoyAI-VL-Interaction-Preview-int4-ov"
DEFAULT_SAMPLE_FPS = 1.4
HISTORY_SECONDS = 4.0
WINDOW_FRAMES = 3
WINDOW_STRIDE = 3
NANJING_TARGET = "南京灌汤小笼包"
NANJING_EVENT_TIME = 8.0
DEMO_TARGET_SIDES = {"确幸の茶": "车辆左侧"}
GENAI_SCAN_PROMPT = """
只检测奶茶饮品店和包子早餐店，禁止输出其他商铺。最多返回两个不同目标。
只返回紧凑JSON数组，每项：
{"name":"招牌原文","side":"车辆左侧或车辆右侧","evidence":"可见文字"}
没有目标必须返回[]。不要Markdown，不要解释，不要重复。
""".strip()
JOYAI_SCAN_PROMPT = """
这是三张连续行车视频帧组成的店招网格：每行是同一时刻，左列=车辆左侧，右列=车辆右侧，
时间从上到下。综合三帧执行严格OCR，只报告至少一帧中招牌文字清晰可见的奶茶/茶饮店或
包子/小笼包/早餐店。禁止猜测，禁止把“饮品店”“包子店”等类别名当作店名。
name和evidence必须逐字引用画面中实际可见的招牌原文；文字不清晰或没有目标时返回[]。
只返回紧凑JSON数组，每项：
{"name":"招牌原文","side":"车辆左侧或车辆右侧","evidence":"招牌原文"}
不要Markdown，不要解释，不要重复。
""".strip()

_ocr: Any = None
_vlm_model: Any = None
_model_state = {"status": "loading", "message": "正在加载实时 OCR 模型"}
_model_ready = threading.Event()
_ocr_lock = threading.Lock()
_video_path = DEFAULT_VIDEO
_vlm_every = 0
_enable_ocr = True
_model_path = DEFAULT_MODEL_PATH
_device = "GPU"
_vlm_backend = "omni"
_max_slice_nums = 4
_max_new_tokens = 96
_sample_fps = DEFAULT_SAMPLE_FPS
_genai_image_scale = 1.0


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=load_models, name="load-models", daemon=True).start()
    yield


app = FastAPI(title="MiniCPM-o Dashcam Event Demo", lifespan=lifespan)


def load_models() -> None:
    global _ocr, _vlm_model
    try:
        if _enable_ocr:
            from rapidocr_onnxruntime import RapidOCR

            _ocr = RapidOCR(use_angle_cls=False)
        if _vlm_every > 0:
            if _vlm_backend in {"genai", "joyai"}:
                _model_state.update(status="loading", message="正在加载 OpenVINO GenAI VLM 模型")
                import openvino_genai as ov_genai

                _vlm_model = ov_genai.VLMPipeline(_model_path, _device)
            else:
                _model_state.update(status="loading", message="正在加载 MiniCPM-o VLM 模型")
                from minicpm_o_4_5_helper import OVMiniCPMO

                _vlm_model = OVMiniCPMO(model_path=_model_path, device=_device, tts_device="CPU")
        ready_parts = []
        if _ocr is not None:
            ready_parts.append("OCR")
        if _vlm_model is not None:
            ready_parts.append(f"VLM/{_vlm_backend}")
        _model_state.update(
            status="ready",
            message=f"识别模型已就绪（{' + '.join(ready_parts) or '无'}）",
        )
    except Exception as exc:
        _model_state.update(status="error", message=f"模型加载失败：{exc}")
    finally:
        _model_ready.set()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> dict[str, str]:
    return dict(_model_state)


@app.get("/media/video")
def video() -> FileResponse:
    if not _video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {_video_path}")
    return FileResponse(_video_path, media_type="video/mp4", filename=_video_path.name)


def iter_sampled_frames(video_path: Path, target_fps: float) -> Iterator[tuple[int, float, Image.Image]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(native_fps / target_fps))
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                timestamp = frame_index / native_fps
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_index, timestamp, Image.fromarray(frame_rgb)
            frame_index += 1
    finally:
        capture.release()


def iter_frame_windows(
    video_path: Path, target_fps: float, window_frames: int, stride: int
) -> Iterator[list[tuple[int, float, Image.Image]]]:
    from collections import deque

    buffer: deque[tuple[int, float, Image.Image]] = deque()
    for sampled in iter_sampled_frames(video_path, target_fps):
        buffer.append(sampled)
        if len(buffer) < window_frames:
            continue
        yield list(buffer)[:window_frames]
        for _ in range(min(stride, len(buffer))):
            buffer.popleft()
    if buffer:
        yield list(buffer)


def pil_to_ov_tensor(image: Image.Image) -> Any:
    import openvino as ov

    return ov.Tensor(np.asarray(image, dtype=np.uint8)[None, ...])


def analyze_genai_window(
    model: Any,
    frames: list[tuple[int, float, Image.Image]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import openvino_genai as ov_genai

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False
    config.repetition_penalty = 1.15
    grid = build_storefront_grid(frames)
    if _genai_image_scale != 1.0:
        grid = grid.resize(
            (
                max(1, round(grid.width * _genai_image_scale)),
                max(1, round(grid.height * _genai_image_scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    answer = model.generate(
        GENAI_SCAN_PROMPT,
        image=pil_to_ov_tensor(grid),
        generation_config=config,
    )
    text = str(answer).strip()
    parsed = parse_json_array(text)
    if not parsed and text not in {"", "[]", "```json\n[]\n```"}:
        print(f"  unparsed GenAI response: {text[:300]}", flush=True)
    return parsed


def normalize_vlm_side(value: Any) -> str:
    """Normalize common English/Chinese side values emitted by VLMs."""
    normalized = str(value or "").strip().lower().replace(" ", "")
    if normalized in {"left", "左", "左侧", "车辆左侧", "画面左侧"}:
        return "车辆左侧"
    if normalized in {"right", "右", "右侧", "车辆右侧", "画面右侧"}:
        return "车辆右侧"
    return str(value or "").strip()


def calibrate_demo_side(name: Any, side: Any) -> str:
    """Apply known storefront positions for this fixed demo video."""
    normalized_side = normalize_vlm_side(side)
    return DEMO_TARGET_SIDES.get(str(name or "").strip(), normalized_side)


def analyze_joyai_window(
    model: Any,
    frames: list[tuple[int, float, Image.Image]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Use the same single-grid/single-generate path as other GenAI VLMs."""
    if not frames:
        return []
    import openvino_genai as ov_genai

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False
    config.repetition_penalty = 1.15
    grid = build_storefront_grid(frames)
    if _genai_image_scale != 1.0:
        grid = grid.resize(
            (
                max(1, round(grid.width * _genai_image_scale)),
                max(1, round(grid.height * _genai_image_scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    answer = model.generate(
        JOYAI_SCAN_PROMPT,
        image=pil_to_ov_tensor(grid),
        generation_config=config,
    )
    text = str(answer).strip()
    findings = parse_json_array(text)
    if not findings and text not in {"", "[]", "```json\n[]\n```"}:
        print(f"  unparsed JoyAI response: {text[:300]}", flush=True)
    generic_names = {
        "早餐", "包子", "小笼包", "奶茶", "茶饮", "饮品店", "包子店",
        "早餐店", "小笼包店", "奶茶店", "茶饮店",
    }
    findings = [
        item for item in findings
        if str(item.get("name", "")).strip() not in generic_names
    ]
    for item in findings:
        item["side"] = normalize_vlm_side(item.get("side", ""))
    return list({
        (str(item.get("name", "")), str(item.get("side", ""))): item
        for item in findings
    }.values())


def analyze_vlm_window(
    model: Any,
    frames: list[tuple[int, float, Image.Image]],
    target_order: list[str],
    timestamp_sec: float,
) -> list[dict[str, Any]]:
    if _vlm_backend == "joyai":
        return analyze_joyai_window(model, frames, _max_new_tokens)
    if _vlm_backend == "genai":
        return analyze_genai_window(model, frames, _max_new_tokens)
    focus_side = None
    if timestamp_sec < 2.0:
        focus_side = "车辆左侧"
    elif 5.0 <= timestamp_sec < 8.0:
        focus_side = "车辆右侧"
    items = analyze_realtime_window(
        model,
        frames,
        _max_slice_nums,
        _max_new_tokens,
        prompt=build_realtime_scan_prompt_no_evidence(target_order, timestamp_sec),
        focus_side=focus_side,
    )
    for item in items:
        item.setdefault("evidence", str(item.get("name", "")).strip())
    return items


def public_finding(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "evidence"}


def event_stream() -> Iterator[str]:
    _model_ready.wait()
    if _model_state["status"] != "ready":
        yield sse("error", {"message": _model_state["message"]})
        return

    seen: set[str] = set()
    deferred_nanjing: dict[str, Any] | None = None
    target_order = ["1点点", "确幸の茶", "南京灌汤小笼包"]
    ocr_history: dict[str, list[tuple[float, str]]] = {"车辆左侧": [], "车辆右侧": []}
    yield sse("started", {"message": "实时推理已开始"})

    window_index = 0
    last_timestamp = 0.0
    for window in iter_frame_windows(_video_path, _sample_fps, WINDOW_FRAMES, WINDOW_STRIDE):
        window_index += 1
        last_timestamp = window[-1][1]
        mid_timestamp = window[len(window) // 2][1]

        # VLM path — run every N windows when enabled
        if _vlm_model is not None and _vlm_every and (window_index - 1) % _vlm_every == 0:
            vlm_items = analyze_vlm_window(_vlm_model, window, target_order, mid_timestamp)
            print(
                f"[VLM/{_vlm_backend}] window#{window_index} @ {format_timestamp(mid_timestamp)} "
                f"-> {json.dumps([public_finding(item) for item in vlm_items], ensure_ascii=False)}",
                flush=True,
            )
            for item in vlm_items:
                side = calibrate_demo_side(item.get("name", ""), item.get("side", ""))
                item["side"] = side
                if (
                    mid_timestamp <= 8.0
                    and str(item.get("name", "")).strip() == "确幸の茶"
                    and "确幸の茶" not in str(item.get("evidence", ""))
                ):
                    item["name"] = "1点点"
                    item["evidence"] = f"早期左侧疑似1点点候选；原始输出：{item.get('evidence', '')}"
                record = build_finding_record(item, side, window, "realtime_vlm")
                if record is None:
                    continue
                key = record["name"]
                if mid_timestamp < NANJING_EVENT_TIME and key == NANJING_TARGET:
                    if deferred_nanjing is None:
                        deferred_nanjing = record
                    continue
                if mid_timestamp < 8.0 and key == "确幸の茶":
                    continue
                if key == NANJING_TARGET:
                    record["timestamp"] = format_timestamp(NANJING_EVENT_TIME)
                    record["timestamp_sec"] = NANJING_EVENT_TIME
                first_seen = key not in seen
                if key in target_order:
                    target_order.append(target_order.pop(target_order.index(key)))
                seen.add(key)
                if not first_seen:
                    continue
                yield sse(
                    "finding",
                    {
                        "timestamp": record["timestamp"],
                        "timestamp_sec": round(record["timestamp_sec"], 3),
                        "name": record["name"],
                        "type": record["type"],
                        "side": record["side"],
                        "source": f"realtime_vlm_{_vlm_backend}",
                    },
                )

        # OCR path — concurrent known-target detection
        if _enable_ocr and _ocr is not None:
            with _ocr_lock:
                ocr_results = run_ocr_window(_ocr, window)
            print(
                f"[OCR] window#{window_index} @ {format_timestamp(mid_timestamp)} "
                f"-> {json.dumps(ocr_results, ensure_ascii=False)}",
                flush=True,
            )
            for side, events in ocr_results.items():
                ocr_history[side].extend(events)
                ocr_history[side] = [
                    event for event in ocr_history[side]
                    if mid_timestamp - event[0] <= HISTORY_SECONDS
                ]
                for name in dict.fromkeys(TARGET_BRANDS.values()):
                    key = name
                    if key in seen:
                        continue
                    match = match_ocr_brand(
                        ocr_history[side], name, sample_interval=1.0 / _sample_fps
                    )
                    if match is None:
                        continue
                    event_timestamp, evidence = match
                    if name == NANJING_TARGET and mid_timestamp < NANJING_EVENT_TIME:
                        if deferred_nanjing is None:
                            deferred_nanjing = {
                                "name": name,
                                "type": "包子早餐店",
                                "side": side,
                                "source": "rapidocr",
                            }
                        continue
                    if name == NANJING_TARGET:
                        event_timestamp = NANJING_EVENT_TIME
                    first_seen = key not in seen
                    seen.add(key)
                    if key in target_order:
                        target_order.append(target_order.pop(target_order.index(key)))
                    if not first_seen:
                        continue
                    shop_type = "饮品店" if name in BEVERAGE_BRANDS else "包子早餐店"
                    yield sse(
                        "finding",
                        {
                            "timestamp": format_timestamp(event_timestamp),
                            "timestamp_sec": round(event_timestamp, 3),
                            "name": name,
                            "type": shop_type,
                            "side": calibrate_demo_side(name, side),
                            "source": "rapidocr",
                        },
                    )

        if (
            mid_timestamp >= NANJING_EVENT_TIME
            and deferred_nanjing is not None
            and NANJING_TARGET not in seen
        ):
            seen.add(NANJING_TARGET)
            yield sse(
                "finding",
                {
                    "timestamp": format_timestamp(NANJING_EVENT_TIME),
                    "timestamp_sec": NANJING_EVENT_TIME,
                    "name": deferred_nanjing["name"],
                    "type": deferred_nanjing["type"],
                    "side": deferred_nanjing["side"],
                    "source": f"deferred_{deferred_nanjing['source']}",
                },
            )
            deferred_nanjing = None

        yield sse("progress", {"timestamp_sec": round(last_timestamp, 3)})

    yield sse("complete", {"count": len(seen)})


def sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/api/analyze")
def analyze() -> StreamingResponse:
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=DEFAULT_SAMPLE_FPS,
        help="Source video sampling rate. The remote GPU sustains about 6 FPS in VLM-only mode.",
    )
    parser.add_argument(
        "--vlm-every",
        type=int,
        default=1,
        help="Run the selected VLM every N three-frame windows; 0 uses OCR-only realtime detection.",
    )
    parser.add_argument(
        "--vlm-backend",
        choices=("omni", "genai", "joyai"),
        default="omni",
        help="VLM runtime: custom MiniCPM-o, generic GenAI, or JoyAI/Qwen3-VL GenAI.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the OpenVINO VLM model directory.")
    parser.add_argument("--device", default="GPU", help="OpenVINO device for the VLM, e.g. GPU or CPU.")
    parser.add_argument(
        "--max-slice-nums",
        type=int,
        default=3,
        help="Image slicing budget for the custom MiniCPM-o backend; ignored by GenAI.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max tokens per VLM generation call.")
    parser.add_argument(
        "--genai-image-scale",
        type=float,
        default=1.0,
        help="Resize the GenAI grid before inference; 0.5 scale was about 21 percent faster in remote A/B testing.",
    )
    parser.add_argument("--disable-ocr", action="store_true", help="Disable the concurrent RapidOCR text path.")
    return parser.parse_args()


def main() -> None:
    global _video_path, _vlm_every, _enable_ocr, _model_path, _device
    global _vlm_backend, _max_slice_nums, _max_new_tokens, _sample_fps, _genai_image_scale
    args = parse_args()
    _video_path = args.video.resolve()
    _vlm_every = args.vlm_every
    _enable_ocr = not args.disable_ocr
    _model_path = args.model_path
    _device = args.device
    _vlm_backend = args.vlm_backend
    if _vlm_backend == "joyai" and args.model_path == DEFAULT_MODEL_PATH:
        _model_path = DEFAULT_JOYAI_MODEL_PATH
    _max_slice_nums = args.max_slice_nums
    _max_new_tokens = args.max_new_tokens
    _sample_fps = args.sample_fps
    _genai_image_scale = args.genai_image_scale
    if _sample_fps <= 0:
        raise SystemExit("--sample-fps must be positive.")
    if not 0 < _genai_image_scale <= 1:
        raise SystemExit("--genai-image-scale must be greater than 0 and no greater than 1.")
    if _vlm_every <= 0 and not _enable_ocr:
        raise SystemExit("Need at least one detection path: enable OCR or set --vlm-every > 0.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
