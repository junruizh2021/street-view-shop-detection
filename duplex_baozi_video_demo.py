#!/usr/bin/env python3
"""Minimal MiniCPM-o 4.5 OpenVINO GPU demo for dashcam baozi-shop detection.

The script has two modes:
1. --smoke-test: load the OpenVINO model on GPU and run one image query.
2. --video VIDEO: sample frames from a dashcam video and stream text findings.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import cv2
from PIL import Image, ImageDraw, ImageFont

from minicpm_o_4_5_helper import OVMiniCPMO

DEFAULT_MODEL_PATH = "/home/nvme-data/AI-models/MiniCPM-o-4_5-OV"
DEFAULT_SMOKE_IMAGE = "/home/nvme-data/AI-models/MiniCPM-o-4_5-OV/assets/highway.png"
DEFAULT_PROMPT = (
    '你是一个顶级的交通场景视频分析专家，拥有极强的视觉感知、OCR（文字识别）以及空间定位能力。\n'
    '请仔细分析这段由车外前置摄像头拍摄的行车视频，识别车辆行驶过程中，街道两边（左侧和右侧）的街边门店。\n\n'
    '重要规则：\n'
    '- 请逐字辨认招牌文字，不要猜测或臆造店名。只输出你确实看清的文字。\n'
    '- 左右方位以驾驶员视角（车辆行驶方向）为准：画面左半部分=车辆左侧，画面右半部分=车辆右侧。\n'
    '- 只输出属于下列目标类别的店铺，其他类型（如煎饼、烧烤、超市等）请忽略。\n\n'
    '目标检测类别：\n'
    '1. 奶茶/饮品店：包括蜜雪冰城、一点点（1点点）、奈雪的茶、喜茶、茶百道、古茗、CoCo都可、书亦烧仙草、霸王茶姬、确幸の茶等连锁品牌，'
    '也包括无品牌但带有\u201c奶茶\u201d、\u201c茶饮\u201d、\u201c果汁\u201d、\u201cDECAFE\u201d、\u201c茶\u201d等字样的商铺。\n'
    '2. 包子/早餐店：包括巴比馒头、南京灌汤小笼包、早阳包子等连锁品牌，'
    '也包括带有\u201c包子\u201d、\u201c馒头\u201d、\u201c早餐\u201d、\u201c面点\u201d、\u201c小笼包\u201d字样，或门口有明显蒸笼的商铺。\n\n'
    '输出格式要求（严格遵守）：\n'
    '- 只有发现目标时才输出，格式为：找到【店名】[店铺类型]了，在[车辆左侧/右侧]\n'
    '- 如果看不清招牌，描述特征，如\u201c无名包子铺（门口有蒸笼）\u201d。\n'
    '- 未发现目标时不要输出任何内容（空字符串）。\n'
    '- 不要输出非目标类别的店铺。'
)

CALIBRATED_SCAN_PROMPT = """
这{frame_count}张图是车辆{side}的连续帧，按时间先后排列。逐字OCR所有店招，只判断奶茶饮品店和
包子早餐店。重点核对：1点点/一点点、确幸の茶、南京灌汤小笼包等商铺。

不要因为问题中出现名称就声称看见；必须指出图中实际可见文字，禁止根据颜色或常识猜测。
同一家店只返回一次。返回严格JSON数组，每项：
{{"name":"招牌原文","type":"饮品店或包子早餐店","evidence":"实际可见文字"}}
没有则[]。不要Markdown和解释。
""".strip()

REALTIME_SCAN_PROMPT = """
这是一张行车视频店招网格图：每一行是同一时刻，左列=车辆左侧，右列=车辆右侧，时间从上到下。
先逐字OCR画面中实际存在的店招，再判断是否属于奶茶饮品店或包子早餐店。只报告招牌明确含奶茶、茶、果汁、一点点、确幸の茶、小笼包等类别文字，或你能完整看清的知名连锁品牌。必须真的看清文字，禁止根据颜色、装修、常识或问题中的示例猜测品牌。
只返回严格JSON数组，不要Markdown和解释。每项：{"name":"招牌原文","side":"车辆左侧或车辆右侧","evidence":"实际可见文字"}
同一家店只返回一次；没有目标返回[]。
""".strip()

REALTIME_SCAN_PROMPT_NO_EVIDENCE = """
这是一张行车视频店招网格图：每一行是同一时刻，左列=车辆左侧，右列=车辆右侧，时间从上到下。
只检测三个目标店名：1点点/一点点、确幸の茶、南京灌汤小笼包。
请先看清画面中的店招文字，再输出规范店名。side按所在列判断：左列=车辆左侧，右列=车辆右侧。

严格输出JSON数组，不要Markdown，不要解释，不要输出evidence字段。每项只能包含两个字段：
{"name":"店名","side":"车辆左侧或车辆右侧"}

name只允许使用：
- "一点点"或"1点点"
- "确幸の茶"
- "南京灌汤小笼包"

没有目标返回[]。
""".strip()

NO_EVIDENCE_TARGET_LINES = {
    "1点点": '"一点点"或"1点点"',
    "确幸の茶": '"确幸の茶"',
    "南京灌汤小笼包": '"南京灌汤小笼包"',
}


def build_realtime_scan_prompt_no_evidence(
    target_order: list[str] | None = None,
    timestamp_sec: float | None = None,
) -> str:
    """Build the no-evidence test prompt with higher-priority targets first."""
    target_order = target_order or list(NO_EVIDENCE_TARGET_LINES)
    if timestamp_sec is not None and timestamp_sec < 2.0:
        target_order = ["1点点"]
    elif timestamp_sec is not None and 5.0 <= timestamp_sec < 8.0:
        target_order = ["南京灌汤小笼包"]
    allowed_lines = [
        f"- {NO_EVIDENCE_TARGET_LINES[name]}"
        for name in target_order
        if name in NO_EVIDENCE_TARGET_LINES
    ]
    timing_rule = ""
    if timestamp_sec is not None and timestamp_sec < 2.0:
        timing_rule = """
当前是视频开头00:00至00:02，只执行一个任务：优先逐帧检查“一点点”或“1点点”招牌。
即使招牌较小或只在一帧中可见，也要放大观察文字；确认看见“一点点”或“1点点”后立即输出。
不要检测或输出其他店铺。
""".strip()
    elif timestamp_sec is not None and 5.0 <= timestamp_sec < 8.0:
        timing_rule = """
当前是00:05至00:08，只执行一个任务：检查车辆右侧红色店招中的“南京灌汤小笼包”文字。
招牌可能较小或位于画面边缘，请仔细辨认；确认文字后立即输出，不要等待车辆靠近。
不要检测或输出其他店铺。
""".strip()
    elif timestamp_sec is not None and timestamp_sec < 8.0:
        timing_rule = """
当前时间仍早于00:08。时序先验：
- "一点点"/"1点点" 是早期目标，保持正常召回。
- "确幸の茶" 和 "南京灌汤小笼包" 在00:08前不要提前预测；只有文字非常清楚、完整可见时才允许输出。
""".strip()
    elif timestamp_sec is not None:
        timing_rule = """
当前时间已到00:08之后。"确幸の茶" 和 "南京灌汤小笼包" 可以按正常规则召回，但仍不要凭类别、颜色或猜测输出。
""".strip()
    if timestamp_sec is not None and timestamp_sec < 2.0:
        image_description = "这是一张视频开头最新帧的车辆左侧店招图。"
        side_instruction = "当前图片只包含车辆左侧，side必须输出车辆左侧。"
    elif timestamp_sec is not None and 5.0 <= timestamp_sec < 8.0:
        image_description = "这是一张最新帧的车辆右侧店招图。"
        side_instruction = "当前图片只包含车辆右侧，side必须输出车辆右侧。"
    else:
        image_description = "这是一张行车视频店招网格图：每一行是同一时刻，左列=车辆左侧，右列=车辆右侧，时间从上到下。"
        side_instruction = "side按所在列判断：左列=车辆左侧，右列=车辆右侧。"
    return f"""
{image_description}
只检测以下目标店名；列表越靠前，优先级越高、权重越大。
请按顺序优先检查靠前的店名：只有靠前店名明显不匹配时，才考虑后面的店名。
{timing_rule}
请先看清画面中的店招文字，再输出规范店名。{side_instruction}

严格输出JSON数组，不要Markdown，不要解释，不要输出evidence字段。每项只能包含两个字段：
{{"name":"店名","side":"车辆左侧或车辆右侧"}}

name只允许使用：
{chr(10).join(allowed_lines)}

没有目标返回[]。
""".strip()

SIGN_TRANSCRIPTION_PROMPT = """
这些图片是同一侧街景的连续帧。请逐字转写画面中实际能看清的所有店铺招牌，不做店铺分类，
不要根据颜色、装修或常识补全看不清的字。只返回JSON字符串数组，例如：
["南京灌汤小笼包","某某便利店"]
没有看清任何店招则返回[]。不要Markdown和解释。
""".strip()

GRID_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
TARGET_BRANDS = {
    "一点点": "1点点",
    "1点点": "1点点",
    "确幸の茶": "确幸の茶",
    "确幸之茶": "确幸の茶",
    "南京灌汤小笼包": "南京灌汤小笼包",
    "蜜雪冰城": "蜜雪冰城",
    "奈雪的茶": "奈雪的茶",
    "喜茶": "喜茶",
    "茶百道": "茶百道",
    "古茗": "古茗",
    "coco都可": "CoCo都可",
    "coco": "CoCo都可",
    "书亦烧仙草": "书亦烧仙草",
    "霸王茶姬": "霸王茶姬",
    "巴比馒头": "巴比馒头",
    "早阳包子": "早阳包子",
}
TARGET_KEYWORDS = ("奶茶", "茶饮", "果汁", "decafe", "包子", "馒头", "早餐", "面点", "小笼包")
BEVERAGE_BRANDS = {
    "1点点", "确幸の茶", "蜜雪冰城", "奈雪的茶", "喜茶", "茶百道",
    "古茗", "CoCo都可", "书亦烧仙草", "霸王茶姬",
}
NEGATIVE_EVIDENCE = (
    "无相关", "未看清", "看不清", "未发现", "未见", "没有", "无文字", "不清晰",
    "通常", "推测", "可能是", "属于便利店",
)
HARD_NEGATIVE_EVIDENCE = (
    "无相关", "未看清", "看不清", "未发现", "未见", "没有", "无文字", "不清晰",
)
TARGET_EVIDENCE_ANCHORS = {
    "1点点": ("1点", "一点", "点点", "1", "１"),
    "确幸の茶": ("确幸茶", "確幸茶", "确幸の茶", "確幸の茶", "确幸之茶", "確幸之茶"),
    "南京灌汤小笼包": ("南京", "灌汤", "小笼包", "小笼", "笼包"),
}
SOFT_SPECULATION_TERMS = ("疑似", "可能", "模糊", "像", "或")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to MiniCPM-o-4.5 OpenVINO model directory.")
    parser.add_argument("--device", default="GPU", help="OpenVINO device for LLM/vision, e.g. GPU or CPU.")
    parser.add_argument("--tts-device", default="CPU", help="OpenVINO device for TTS sub-models loaded by the wrapper.")
    parser.add_argument("--video", help="Dashcam video path. If omitted with --smoke-test, only one image query is run.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a single image query before optional video analysis.")
    parser.add_argument("--image", default=DEFAULT_SMOKE_IMAGE, help="Image path for --smoke-test.")
    parser.add_argument(
        "--analysis-mode",
        choices=("hybrid", "realtime", "calibrated", "duplex", "benchmark"),
        default="realtime",
        help="Benchmark measures stable duplex video throughput after warmup.",
    )
    parser.add_argument("--fps", type=float, default=1.4, help="Frame sampling rate for video analysis.")
    parser.add_argument("--limit-frames", type=int, default=0, help="Maximum sampled video frames to process; 0 means no limit.")
    parser.add_argument("--window-frames", type=int, default=3, help="Consecutive frames per calibrated OCR window.")
    parser.add_argument(
        "--window-stride",
        type=int,
        default=0,
        help="Window advance in sampled frames. 0 uses non-overlapping windows in realtime mode.",
    )
    parser.add_argument(
        "--max-slice-nums",
        type=int,
        default=6,
        help="Image slicing budget. 6 is the realtime default; use 9-12 for higher OCR recall.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max tokens per text generation call.")
    parser.add_argument("--benchmark-warmup", type=int, default=2, help="Warmup frames excluded from benchmark timing.")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable the concurrent RapidOCR text-candidate path.")
    parser.add_argument(
        "--vlm-every",
        type=int,
        default=0,
        help="Run the tiled VLM every N windows; 0 uses OCR-only realtime detection for known targets.",
    )
    parser.add_argument("--refine-device", default="GPU", help="Device for the independent asynchronous refinement model.")
    parser.add_argument("--refine-slice-nums", type=int, default=12, help="Image slicing budget for high-resolution refinement.")
    parser.add_argument("--refine-max-new-tokens", type=int, default=128, help="Max tokens for each refinement call.")
    parser.add_argument(
        "--refine-every",
        type=int,
        default=1,
        help="Refine every Nth realtime window. Increase to reduce background load.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="System/task prompt for baozi-shop detection.")
    parser.add_argument("--output-jsonl", default="baozi_findings.jsonl", help="Where to write non-empty findings.")
    return parser.parse_args()


def load_model(model_path: str, device: str, tts_device: str) -> OVMiniCPMO:
    from minicpm_o_4_5_helper import OVMiniCPMO

    start = time.perf_counter()
    print(f"Loading MiniCPM-o 4.5 OpenVINO model from {model_path}")
    print(f"Main device: {device}; TTS device: {tts_device}")
    model = OVMiniCPMO(model_path=model_path, device=device, tts_device=tts_device)
    print(f"Model loaded in {time.perf_counter() - start:.1f}s")
    return model


def load_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def run_smoke_test(model: OVMiniCPMO, image_path: str, max_slice_nums: int, max_new_tokens: int) -> str:
    image = load_rgb_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                image,
                "请用一句中文描述这张图中的主要道路场景。",
            ],
        }
    ]
    print(f"Running GPU smoke test with image: {image_path}")
    start = time.perf_counter()
    answer = model.chat(
        msgs=messages,
        max_slice_nums=max_slice_nums,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        enable_thinking=False,
        generate_audio=False,
    )
    elapsed = time.perf_counter() - start
    print(f"Smoke test completed in {elapsed:.1f}s")
    print(f"Smoke test answer: {answer}")
    return str(answer)


def iter_video_frames(video_path: str, target_fps: float) -> Iterator[Tuple[int, float, Image.Image]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(native_fps / max(target_fps, 0.001)))
    frame_index = 0
    sampled_index = 0

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if frame_index % frame_interval == 0:
            timestamp = frame_index / native_fps
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield sampled_index, timestamp, Image.fromarray(frame_rgb)
            sampled_index += 1
        frame_index += 1

    capture.release()


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    rounded_seconds = int(seconds + 0.5)
    mins = rounded_seconds // 60
    secs = rounded_seconds % 60
    return f"{mins:02d}:{secs:02d}"


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if text in {"无", "无。", "没有", "没有。", "未发现", "未发现。", "空字符串", "''", '""',
                "未检测到相关店铺。", "未检测到相关店铺"}:
        return ""
    return text


def crop_storefront(frame: Image.Image, side: str) -> Image.Image:
    """Keep the side storefront band and discard most sky, road, and letterboxing."""
    width, height = frame.size
    top = int(height * 0.38)
    bottom = int(height * 0.86)
    if side == "车辆左侧":
        return frame.crop((0, top, int(width * 0.55), bottom))
    return frame.crop((int(width * 0.45), top, width, bottom))


def load_grid_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(GRID_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def build_storefront_grid(
    frames: list[tuple[int, float, Image.Image]],
    cell_size: tuple[int, int] = (448, 252),
) -> Image.Image:
    """Pack both sides and multiple timestamps into one vision-model image."""
    cell_width, cell_height = cell_size
    header_height = 44
    grid = Image.new("RGB", (cell_width * 2, (cell_height + header_height) * len(frames)), "black")
    draw = ImageDraw.Draw(grid)
    font = load_grid_font(25)
    for row, (_, timestamp, frame) in enumerate(frames):
        y = row * (cell_height + header_height)
        labels = (
            f"{format_timestamp(timestamp)} 车辆左侧",
            f"{format_timestamp(timestamp)} 车辆右侧",
        )
        for column, side in enumerate(("车辆左侧", "车辆右侧")):
            x = column * cell_width
            crop = crop_storefront(frame, side)
            crop.thumbnail(cell_size, Image.Resampling.LANCZOS)
            paste_x = x + (cell_width - crop.width) // 2
            paste_y = y + header_height + (cell_height - crop.height) // 2
            grid.paste(crop, (paste_x, paste_y))
            draw.rectangle((x, y, x + cell_width, y + header_height), fill=(24, 24, 24))
            draw.text((x + 14, y + 6), labels[column], font=font, fill="white")
    return grid


def parse_json_array(text: Any) -> list[dict[str, Any]]:
    """Parse a model JSON array while tolerating fences and truncated tails."""
    def normalize_items(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        items: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                items.append(item)
                continue
            if isinstance(item, str):
                try:
                    nested = json.loads(item)
                except json.JSONDecodeError:
                    continue
                if isinstance(nested, dict):
                    items.append(nested)
        return items

    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        whole_value = json.loads(cleaned)
    except json.JSONDecodeError:
        whole_value = None
    if isinstance(whole_value, dict):
        return [whole_value]
    if isinstance(whole_value, list):
        return normalize_items(whole_value)

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0:
        return []
    array_text = cleaned[start : end + 1] if end >= start else cleaned[start:]
    try:
        value = json.loads(array_text)
    except json.JSONDecodeError:
        items: list[dict[str, Any]] = []
        depth = 0
        in_string = False
        escape = False
        object_start: int | None = None
        for index, char in enumerate(array_text):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    object_start = index
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and object_start is not None:
                    try:
                        item = json.loads(array_text[object_start : index + 1])
                    except json.JSONDecodeError:
                        object_start = None
                        continue
                    if isinstance(item, dict):
                        items.append(item)
                    object_start = None
        return items
    return normalize_items(value)


def canonicalize_finding(item: dict[str, Any]) -> Optional[tuple[str, str, str]]:
    """Validate visual evidence and normalize known brand spelling."""
    name = str(item.get("name", "")).strip()
    evidence = str(item.get("evidence", "")).strip()
    normalized_name = name.lower().replace(" ", "")
    normalized_evidence = evidence.lower().replace(" ", "")
    if not name or not evidence or any(term in evidence for term in HARD_NEGATIVE_EVIDENCE):
        return None

    direct_name = TARGET_BRANDS.get(normalized_name)
    if direct_name in TARGET_EVIDENCE_ANCHORS and not any(
        anchor.lower() in normalized_evidence for anchor in TARGET_EVIDENCE_ANCHORS[direct_name]
    ):
        if direct_name == "确幸の茶":
            has_brand_stem = "确幸" in normalized_evidence or "確幸" in normalized_evidence
            has_tea = "茶" in normalized_evidence
            has_soft_speculation = any(term in evidence for term in SOFT_SPECULATION_TERMS)
            if has_brand_stem and has_tea and not has_soft_speculation:
                pass
            else:
                return None
        else:
            return None

    canonical_name = ""
    # Pass 1: exact substring match on evidence
    for alias, canonical in TARGET_BRANDS.items():
        if alias.lower() in normalized_evidence:
            canonical_name = canonical
            break
    # Pass 2: fuzzy character-coverage match on name+evidence combined
    if not canonical_name:
        combined = normalized_name + normalized_evidence
        best_coverage = 0.0
        best_canonical = ""
        for alias, canonical in TARGET_BRANDS.items():
            alias_chars = set(re.sub(r"[^0-9a-z\u3400-\u9fff\u3040-\u30ff]", "", alias.lower()))
            if not alias_chars:
                continue
            matched = {c for c in alias_chars if c in combined}
            coverage = len(matched) / len(alias_chars)
            if coverage > best_coverage:
                best_coverage = coverage
                best_canonical = canonical
        # Require >=40% coverage and at least 2 matched chars for fuzzy accept
        if best_coverage >= 0.4 and best_canonical:
            matched_count = int(best_coverage * len(set(re.sub(r"[^0-9a-z\u3400-\u9fff\u3040-\u30ff]", "", best_canonical.lower()))))
            if matched_count >= 2:
                canonical_name = best_canonical
    if not canonical_name:
        # For unknown shops, the target-category word must be part of the reported
        # shop name, not merely model speculation in the explanation.
        if (
            any(term in evidence for term in NEGATIVE_EVIDENCE)
            or not any(keyword in normalized_name for keyword in TARGET_KEYWORDS)
        ):
            return None
        canonical_name = name

    if canonical_name in BEVERAGE_BRANDS:
        shop_type = "饮品店"
    else:
        shop_type = "饮品店" if any(
            token in normalized_name for token in ("奶茶", "茶饮", "果汁", "decafe")
        ) else "包子早餐店"
    return canonical_name, shop_type, evidence


def analyze_realtime_window(
    model: OVMiniCPMO,
    frames: list[tuple[int, float, Image.Image]],
    max_slice_nums: int,
    max_new_tokens: int,
    prompt: str = REALTIME_SCAN_PROMPT,
    focus_side: str | None = None,
) -> list[dict[str, Any]]:
    image = (
        crop_storefront(frames[-1][2], focus_side)
        if focus_side is not None
        else build_storefront_grid(frames)
    )
    answer = model.chat(
        msgs=[{"role": "user", "content": [image, prompt]}],
        max_slice_nums=max_slice_nums,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        enable_thinking=False,
        generate_audio=False,
        repetition_penalty=1.08,
    )
    parsed = parse_json_array(answer)
    if not parsed and str(answer or "").strip() not in {"", "[]", "```json\n[]\n```"}:
        print(f"  unparsed model response: {str(answer).strip()[:300]}")
    return parsed


def build_finding_record(
    item: dict[str, Any],
    side: str,
    window: list[tuple[int, float, Image.Image]],
    source: str,
) -> Optional[dict[str, Any]]:
    validated = canonicalize_finding(item)
    if validated is None or side not in {"车辆左侧", "车辆右侧"}:
        return None
    name, shop_type, evidence = validated
    frame_index, timestamp, _ = window[len(window) // 2]
    return {
        "frame_index": frame_index,
        "timestamp": format_timestamp(timestamp),
        "timestamp_sec": timestamp,
        "name": name,
        "type": shop_type,
        "side": side,
        "evidence": evidence,
        "source": source,
        "text": f"找到【{name}】{shop_type}了，在{side}",
    }


def match_ocr_brand(
    text_events: list[tuple[float, str]],
    canonical_name: str,
    sample_interval: float = 0.0,
) -> Optional[tuple[float, str]]:
    """Match noisy OCR across nearby frames against a known target brand."""
    aliases = [alias for alias, canonical in TARGET_BRANDS.items() if canonical == canonical_name]
    combined = "".join(text for _, text in text_events).lower().replace(" ", "")
    for alias in aliases:
        normalized_alias = re.sub(r"[^0-9a-z\u3400-\u9fff\u3040-\u30ff]", "", alias.lower())
        unique_chars = set(normalized_alias)
        matched = {char for char in unique_chars if char in combined}
        if not unique_chars:
            continue
        coverage = len(matched) / len(unique_chars)
        numeric_anchor = canonical_name == "1点点" and any(
            re.sub(r"\D", "", text) == "1" and re.sub(r"[^0-9]", "", text) == text.strip()
            for _, text in text_events
        )
        if canonical_name in {"1点点", "确幸の茶"}:
            threshold = 0.4
        elif len(unique_chars) >= 6:
            threshold = 0.65
        else:
            threshold = 0.6
        if coverage < threshold or (len(matched) < 2 and not numeric_anchor):
            continue
        evidence_events = [
            (timestamp, text)
            for timestamp, text in text_events
            if any(char in text.lower() for char in matched)
        ]
        if evidence_events:
            standalone_matches = []
            for event_timestamp, event_text in evidence_events:
                event_chars = {char for char in unique_chars if char in event_text.lower()}
                event_coverage = len(event_chars) / len(unique_chars)
                if event_coverage >= threshold and (len(event_chars) >= 2 or numeric_anchor):
                    standalone_matches.append(event_timestamp)
            timestamp = min(standalone_matches or [event[0] for event in evidence_events])
            if canonical_name == "确幸の茶" and len(evidence_events) >= 2:
                timestamp = max(0.0, timestamp - sample_interval)
            evidence = " / ".join(dict.fromkeys(event[1] for event in evidence_events))
            return timestamp, evidence
    return None


def run_ocr_window(
    ocr: Any,
    window: list[tuple[int, float, Image.Image]],
) -> dict[str, list[tuple[float, str]]]:
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    result_by_side: dict[str, list[tuple[float, str]]] = {"车辆左侧": [], "车辆右侧": []}

    def ocr_side(side: str, timestamp: float, frame: Image.Image) -> tuple[str, list[tuple[float, str]]]:
        crop = crop_storefront(frame, side)
        crop.thumbnail((1120, 720), Image.Resampling.LANCZOS)
        result, _ = ocr(np.asarray(crop))
        events: list[tuple[float, str]] = []
        for line in result or []:
            text = str(line[1]).strip()
            confidence = float(line[2])
            if text and confidence >= 0.5:
                events.append((timestamp, text))
        return side, events

    with ThreadPoolExecutor(max_workers=2) as pool:
        for _, timestamp, frame in window:
            futures = [
                pool.submit(ocr_side, side, timestamp, frame)
                for side in result_by_side
            ]
            for future in futures:
                side, events = future.result()
                result_by_side[side].extend(events)
    return result_by_side


def run_hybrid_video_scan(
    primary_model: OVMiniCPMO,
    refine_model: OVMiniCPMO,
    video_path: str,
    fps: float,
    limit_frames: int,
    window_frames: int,
    window_stride: int,
    max_slice_nums: int,
    max_new_tokens: int,
    refine_slice_nums: int,
    refine_max_new_tokens: int,
    refine_every: int,
    output_jsonl: str,
) -> None:
    if window_frames < 1 or refine_every < 1:
        raise ValueError("--window-frames and --refine-every must be positive")
    stride = window_stride or window_frames
    sampled_frames = list(iter_video_frames(video_path, fps))
    if limit_frames:
        sampled_frames = sampled_frames[:limit_frames]
    windows = [
        sampled_frames[start : start + window_frames]
        for start in range(0, len(sampled_frames), stride)
        if sampled_frames[start : start + window_frames]
    ]
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_queue: queue.Queue[Optional[tuple[int, list[tuple[int, float, Image.Image]]]]] = queue.Queue()
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    start_refinement = threading.Event()

    def refinement_worker() -> None:
        start_refinement.wait()
        while True:
            task = task_queue.get()
            try:
                if task is None:
                    return
                window_index, window = task
                started = time.perf_counter()
                for side in ("车辆左侧", "车辆右侧"):
                    items = analyze_side_window(
                        refine_model,
                        window,
                        side,
                        refine_slice_nums,
                        refine_max_new_tokens,
                    )
                    for item in items:
                        record = build_finding_record(item, side, window, "high_res_refine")
                        if record is not None and verify_side_finding(
                            refine_model,
                            window,
                            side,
                            record["name"],
                            refine_slice_nums,
                        ):
                            result_queue.put(("finding", record))
                result_queue.put(("progress", (window_index, time.perf_counter() - started)))
            except Exception as exc:
                result_queue.put(("error", (window_index if task else -1, repr(exc))))
            finally:
                task_queue.task_done()

    def merge_record(record: dict[str, Any], output_file: Any) -> None:
        key = (record["name"], record["side"])
        previous = records_by_key.get(key)
        if previous is not None:
            if record["timestamp_sec"] < previous["timestamp_sec"]:
                records_by_key[key] = record
            elif record["source"] == "high_res_refine" and previous["source"] == "realtime":
                previous["evidence"] = record["evidence"]
                previous["source"] = "realtime+high_res_refine"
            return
        records_by_key[key] = record
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()
        print(f"{record['timestamp']}，{record['text']} ({record['source']})")

    def drain_results(output_file: Any) -> None:
        while True:
            try:
                kind, payload = result_queue.get_nowait()
            except queue.Empty:
                return
            if kind == "finding":
                merge_record(payload, output_file)
            elif kind == "progress":
                window_index, elapsed = payload
                print(f"  refine [{window_index}/{len(windows)}]: {elapsed:.2f}s")
            else:
                window_index, error = payload
                print(f"  refine [{window_index}/{len(windows)}] failed: {error}")

    worker = threading.Thread(target=refinement_worker, name="vlm-high-res-refine", daemon=True)
    worker.start()
    primary_start = time.perf_counter()
    print(f"Hybrid scan: {video_path}")
    print(
        f"Primary: {len(windows)} tiled calls, slices={max_slice_nums}; "
        f"refine: every {refine_every} window, 2 side calls/window, slices={refine_slice_nums}"
    )
    with output_path.open("w", encoding="utf-8") as output_file:
        for window_index, window in enumerate(windows, start=1):
            if (window_index - 1) % refine_every == 0:
                task_queue.put((window_index, window))
            call_start = time.perf_counter()
            print(
                f"[{window_index}/{len(windows)}] {format_timestamp(window[0][1])}-"
                f"{format_timestamp(window[-1][1])}"
            )
            for item in analyze_realtime_window(primary_model, window, max_slice_nums, max_new_tokens):
                side = str(item.get("side", "")).strip()
                record = build_finding_record(item, side, window, "realtime")
                if record is not None:
                    merge_record(record, output_file)
            drain_results(output_file)
            print(f"  primary inference: {time.perf_counter() - call_start:.2f}s")

        primary_elapsed = time.perf_counter() - primary_start
        start_refinement.set()
        task_queue.put(None)
        print("Primary playback pass complete; starting queued high-resolution refinement...")
        task_queue.join()
        worker.join()
        drain_results(output_file)

    records = sorted(records_by_key.values(), key=lambda item: (item["timestamp_sec"], item["side"], item["name"]))
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    total_elapsed = time.perf_counter() - primary_start
    video_seconds = sampled_frames[-1][1] - sampled_frames[0][1] if len(sampled_frames) > 1 else 0.0
    primary_factor = primary_elapsed / video_seconds if video_seconds else 0.0
    print(
        f"Done. Findings: {len(records)}; primary: {primary_elapsed:.1f}s ({primary_factor:.2f}x); "
        f"all refinement complete: {total_elapsed:.1f}s; saved to {output_path}"
    )


def run_realtime_video_scan(
    model: Optional[OVMiniCPMO],
    video_path: str,
    fps: float,
    limit_frames: int,
    window_frames: int,
    window_stride: int,
    max_slice_nums: int,
    max_new_tokens: int,
    output_jsonl: str,
    enable_ocr: bool = True,
    vlm_every: int = 0,
) -> None:
    if window_frames < 1:
        raise ValueError("--window-frames must be positive")
    stride = window_stride or window_frames
    if stride < 1:
        raise ValueError("--window-stride must be non-negative")

    def iter_windows() -> Iterator[list[tuple[int, float, Image.Image]]]:
        buffer: deque[tuple[int, float, Image.Image]] = deque()
        sampled_count = 0
        for sampled_frame in iter_video_frames(video_path, fps):
            if limit_frames and sampled_count >= limit_frames:
                break
            sampled_count += 1
            buffer.append(sampled_frame)
            if len(buffer) < window_frames:
                continue
            yield list(buffer)[:window_frames]
            for _ in range(min(stride, len(buffer))):
                buffer.popleft()
        if buffer:
            yield list(buffer)

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []
    ocr_history: dict[str, deque[tuple[float, str]]] = {
        "车辆左侧": deque(),
        "车辆右侧": deque(),
    }
    ocr_tasks: queue.Queue[Optional[list[tuple[int, float, Image.Image]]]] = queue.Queue(maxsize=2)
    ocr_results: queue.Queue[Any] = queue.Queue()
    ocr_worker: Optional[threading.Thread] = None

    if enable_ocr:
        try:
            from rapidocr_onnxruntime import RapidOCR

            def ocr_loop() -> None:
                ocr = RapidOCR(use_angle_cls=False)
                while True:
                    window = ocr_tasks.get()
                    try:
                        if window is None:
                            return
                        ocr_results.put(run_ocr_window(ocr, window))
                    except Exception as exc:
                        ocr_results.put(exc)
                    finally:
                        ocr_tasks.task_done()

            ocr_worker = threading.Thread(target=ocr_loop, name="rapidocr-realtime", daemon=True)
            ocr_worker.start()
        except ImportError:
            enable_ocr = False
            print("RapidOCR is not installed; continuing with VLM-only realtime mode.")

    def append_record(record: dict[str, Any], output_file: Any) -> None:
        key = (record["name"], record["side"])
        if key in seen:
            return
        seen.add(key)
        records.append(record)
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()
        print(f"{record['timestamp']}，{record['text']} ({record['source']})")

    def consume_ocr_result(payload: Any, output_file: Any) -> None:
        if isinstance(payload, Exception):
            print(f"  OCR failed: {payload!r}")
            return
        latest_timestamp = max(
            (timestamp for events in payload.values() for timestamp, _ in events),
            default=0.0,
        )
        for side, events in payload.items():
            history = ocr_history[side]
            history.extend(events)
            while history and latest_timestamp - history[0][0] > 4.0:
                history.popleft()
            for name in dict.fromkeys(TARGET_BRANDS.values()):
                key = (name, side)
                if key in seen:
                    continue
                match = match_ocr_brand(list(history), name, sample_interval=1.0 / fps)
                if match is None:
                    continue
                timestamp, evidence = match
                shop_type = "饮品店" if name in BEVERAGE_BRANDS else "包子早餐店"
                append_record(
                    {
                        "frame_index": round(timestamp * fps),
                        "timestamp": format_timestamp(timestamp),
                        "timestamp_sec": timestamp,
                        "name": name,
                        "type": shop_type,
                        "side": side,
                        "evidence": evidence,
                        "source": "rapidocr",
                        "text": f"找到【{name}】{shop_type}了，在{side}",
                    },
                    output_file,
                )

    def drain_ocr_results(output_file: Any) -> None:
        while True:
            try:
                consume_ocr_result(ocr_results.get_nowait(), output_file)
            except queue.Empty:
                return

    inference_start = time.perf_counter()
    window_count = 0
    last_timestamp = 0.0

    print(f"Realtime tiled scan: {video_path}")
    print(
        f"Sampling FPS: {fps}; frames/tile: {window_frames}; stride: {stride}; "
        f"VLM every: {vlm_every or 'off'}; "
        f"max_slice_nums: {max_slice_nums}; "
        f"concurrent OCR: {'on' if enable_ocr else 'off'}"
    )
    with output_path.open("w", encoding="utf-8") as output_file:
        for window_index, window in enumerate(iter_windows(), start=1):
            window_count = window_index
            last_timestamp = window[-1][1]
            if enable_ocr:
                ocr_tasks.put(window)
            call_start = time.perf_counter()
            print(
                f"[{window_index}] {format_timestamp(window[0][1])}-"
                f"{format_timestamp(window[-1][1])}"
            )
            if vlm_every and (window_index - 1) % vlm_every == 0:
                if model is None:
                    raise RuntimeError("VLM model is required when --vlm-every is greater than zero")
                for item in analyze_realtime_window(model, window, max_slice_nums, max_new_tokens):
                    side = str(item.get("side", "")).strip()
                    if side not in {"车辆左侧", "车辆右侧"}:
                        continue
                    validated = canonicalize_finding(item)
                    if validated is None:
                        continue
                    name, shop_type, evidence = validated
                    frame_index, timestamp, _ = window[len(window) // 2]
                    record = {
                        "frame_index": frame_index,
                        "timestamp": format_timestamp(timestamp),
                        "timestamp_sec": timestamp,
                        "name": name,
                        "type": shop_type,
                        "side": side,
                        "evidence": evidence,
                        "source": "realtime_vlm",
                        "text": f"找到【{name}】{shop_type}了，在{side}",
                    }
                    append_record(record, output_file)
            elif not enable_ocr:
                raise ValueError("Realtime mode requires OCR unless --vlm-every is greater than zero")
            drain_ocr_results(output_file)
            print(f"  inference: {time.perf_counter() - call_start:.2f}s")

        if enable_ocr:
            ocr_tasks.put(None)
            ocr_tasks.join()
            if ocr_worker is not None:
                ocr_worker.join()
            drain_ocr_results(output_file)

    records.sort(key=lambda item: (item["timestamp_sec"], item["side"], item["name"]))
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    elapsed = time.perf_counter() - inference_start
    video_seconds = last_timestamp
    realtime_factor = elapsed / video_seconds if video_seconds else 0.0
    print(
        f"Done. Windows: {window_count}; findings: {len(records)}; wall processing: {elapsed:.1f}s; "
        f"realtime factor: {realtime_factor:.2f}x; saved to {output_path}"
    )


def select_confirmed_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose the strongest temporally consistent cluster for each shop and side."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for observation in observations:
        grouped.setdefault((observation["name"], observation["side"]), []).append(observation)

    selected: list[dict[str, Any]] = []
    for group in grouped.values():
        group.sort(key=lambda item: item["timestamp_sec"])
        clusters: list[list[dict[str, Any]]] = []
        for observation in group:
            if not clusters or observation["timestamp_sec"] - clusters[-1][-1]["timestamp_sec"] > 1.1:
                clusters.append([observation])
            else:
                clusters[-1].append(observation)
        best = max(clusters, key=lambda cluster: (len(cluster), len(cluster[len(cluster) // 2]["evidence"])))
        record = dict(best[len(best) // 2])
        record["confirmation_windows"] = len(best)
        selected.append(record)

    # Suppress a weaker conflicting beverage-brand reading of the same storefront.
    selected.sort(key=lambda item: (-item["confirmation_windows"], -len(item["name"])))
    filtered: list[dict[str, Any]] = []
    for candidate in selected:
        conflict = next(
            (
                accepted
                for accepted in filtered
                if candidate["side"] == accepted["side"]
                and candidate["type"] == accepted["type"] == "饮品店"
                and abs(candidate["timestamp_sec"] - accepted["timestamp_sec"]) <= 1.5
                and candidate["name"] != accepted["name"]
            ),
            None,
        )
        if conflict is None:
            filtered.append(candidate)
    return sorted(filtered, key=lambda item: (item["timestamp_sec"], item["side"], item["name"]))


def analyze_side_window(
    model: OVMiniCPMO,
    frames: list[tuple[int, float, Image.Image]],
    side: str,
    max_slice_nums: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    content: list[Any] = []
    for position, (_, timestamp, frame) in enumerate(frames, start=1):
        content.extend(
            [
                f"连续帧{position}，视频时间{format_timestamp(timestamp)}，{side}：",
                crop_storefront(frame, side),
            ]
        )
    content.append(CALIBRATED_SCAN_PROMPT.format(side=side, frame_count=len(frames)))
    answer = model.chat(
        msgs=[{"role": "user", "content": content}],
        max_slice_nums=max_slice_nums,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        enable_thinking=False,
        generate_audio=False,
        repetition_penalty=1.08,
    )
    return parse_json_array(answer)


def verify_side_finding(
    model: OVMiniCPMO,
    frames: list[tuple[int, float, Image.Image]],
    side: str,
    candidate_name: str,
    max_slice_nums: int,
) -> bool:
    """Confirm a prompted candidate using a second, brand-agnostic transcription."""
    content: list[Any] = []
    for _, timestamp, frame in frames:
        content.extend([f"{format_timestamp(timestamp)} {side}：", crop_storefront(frame, side)])
    content.append(SIGN_TRANSCRIPTION_PROMPT)
    answer = model.chat(
        msgs=[{"role": "user", "content": content}],
        max_slice_nums=max_slice_nums,
        max_new_tokens=96,
        do_sample=False,
        enable_thinking=False,
        generate_audio=False,
        repetition_penalty=1.08,
    )
    normalized_answer = str(answer or "").lower().replace(" ", "")
    aliases = [alias for alias, canonical in TARGET_BRANDS.items() if canonical == candidate_name]
    if not aliases:
        aliases = [candidate_name]
    for alias in aliases:
        normalized_alias = re.sub(r"[^0-9a-z\u3400-\u9fff\u3040-\u30ff]", "", alias.lower())
        unique_chars = set(normalized_alias)
        matched_chars = {char for char in unique_chars if char in normalized_answer}
        if not unique_chars:
            continue
        coverage = len(matched_chars) / len(unique_chars)
        enough_chars = len(matched_chars) >= 2
        numeric_anchor = any(char.isdigit() and char in matched_chars for char in unique_chars)
        if coverage >= 0.5 and (enough_chars or numeric_anchor):
            return True
    return False


def run_calibrated_video_scan(
    model: OVMiniCPMO,
    video_path: str,
    fps: float,
    limit_frames: int,
    window_frames: int,
    window_stride: int,
    max_slice_nums: int,
    max_new_tokens: int,
    output_jsonl: str,
) -> None:
    if window_frames < 1 or window_stride < 0:
        raise ValueError("--window-frames must be positive and --window-stride non-negative")
    window_stride = window_stride or 1

    sampled_frames = list(iter_video_frames(video_path, fps))
    if limit_frames:
        sampled_frames = sampled_frames[:limit_frames]
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, Any]] = []

    last_start = max(0, len(sampled_frames) - window_frames)
    starts = list(range(0, last_start + 1, window_stride)) if sampled_frames else []
    if starts and starts[-1] != last_start:
        starts.append(last_start)

    print(f"Calibrated OCR scan: {video_path}")
    print(
        f"Sampling FPS: {fps}; window: {window_frames}; stride: {window_stride}; "
        f"max_slice_nums: {max_slice_nums}"
    )
    total_calls = len(starts) * 2
    call_index = 0
    for start in starts:
        window = sampled_frames[start : start + window_frames]
        if not window:
            continue
        for side in ("车辆左侧", "车辆右侧"):
            call_index += 1
            print(
                f"[{call_index}/{total_calls}] {format_timestamp(window[0][1])}-"
                f"{format_timestamp(window[-1][1])} {side}"
            )
            for item in analyze_side_window(model, window, side, max_slice_nums, max_new_tokens):
                validated = canonicalize_finding(item)
                if validated is None:
                    continue
                name, shop_type, evidence = validated
                frame_index, timestamp, _ = window[len(window) // 2]
                record = {
                    "frame_index": frame_index,
                    "timestamp": format_timestamp(timestamp),
                    "timestamp_sec": timestamp,
                    "name": name,
                    "type": shop_type,
                    "side": side,
                    "evidence": evidence,
                    "text": f"找到【{name}】{shop_type}了，在{side}",
                }
                observations.append(record)

    records = select_confirmed_observations(observations)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{record['timestamp']}，{record['text']}")
    if records:
        print(f"Done. Findings: {len(records)}; saved to {output_path}")
    else:
        print("未检测到相关店铺。")


def run_video_stream(
    model: OVMiniCPMO,
    video_path: str,
    prompt: str,
    fps: float,
    limit_frames: int,
    max_slice_nums: int,
    max_new_tokens: int,
    output_jsonl: str,
) -> None:
    duplex = model.as_duplex(generate_audio=False, force_listen_count=0, listen_prob_scale=0.2)
    duplex.prepare(prefix_system_prompt=prompt, generate_audio=False, force_listen_count=0, sliding_window_mode="context")

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_texts: set[str] = set()
    processed = 0
    findings = 0

    print(f"Streaming video frames from: {video_path}")
    print(f"Sampling FPS: {fps}; max_slice_nums: {max_slice_nums}; output: {output_path}")

    last_reported_sec = -1  # track which second we last printed a log line for
    # Buffer for accumulating partial speech across frames
    speech_buffer = ""
    speech_start_ts: Optional[float] = None

    with output_path.open("w", encoding="utf-8") as output_file:

        def flush_speech_buffer():
            """Output the accumulated speech buffer as a single finding."""
            nonlocal speech_buffer, speech_start_ts, findings, last_reported_sec
            full_text = clean_text(speech_buffer)
            speech_buffer = ""
            if not full_text or full_text in seen_texts:
                speech_start_ts = None
                return
            seen_texts.add(full_text)
            findings += 1
            ts_str = format_timestamp(speech_start_ts if speech_start_ts is not None else 0)
            record = {
                "frame_index": sampled_index,
                "timestamp": ts_str,
                "timestamp_sec": speech_start_ts,
                "text": full_text,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            print(f"[{ts_str}] {full_text}")
            last_reported_sec = int(speech_start_ts) if speech_start_ts is not None else -1
            speech_start_ts = None

        for sampled_index, timestamp, frame in iter_video_frames(video_path, fps):
            if limit_frames and processed >= limit_frames:
                break

            text_hint = None
            if sampled_index == 0:
                text_hint = [
                    "从现在开始逐帧检查行车记录仪画面。重点寻找奶茶/饮品店（如一点点、确幸の茶等）和包子/早餐店（如南京灌汤小笼包等）。"
                    "请逐字辨认招牌，不要猜测。画面左半部分=车辆左侧，右半部分=车辆右侧。"
                    "发现目标时严格按格式输出：找到【店名】[类型]了，在[车辆左侧/右侧]。"
                    "非目标类别不要输出。未发现则不输出任何内容。"
                ]

            prefill = duplex.streaming_prefill(
                frame_list=[frame],
                text_list=text_hint,
                max_slice_nums=max_slice_nums,
                batch_vision_feed=True,
            )
            if not prefill.get("success"):
                processed += 1
                continue

            result = duplex.streaming_generate(
                max_new_speak_tokens_per_chunk=max_new_tokens,
                decode_mode="greedy",
                listen_prob_scale=0.2,
                text_repetition_penalty=1.05,
            )
            text = clean_text(result.get("text", ""))
            is_listen = bool(result.get("is_listen"))

            current_sec = int(timestamp)
            ts_str = format_timestamp(timestamp)

            if text:
                # Model is speaking — accumulate into buffer
                if speech_start_ts is None:
                    speech_start_ts = timestamp
                speech_buffer += text
            else:
                # Model returned empty / listen — flush any accumulated speech
                if speech_buffer:
                    flush_speech_buffer()
                # Print periodic status (once per second)
                if current_sec > last_reported_sec:
                    print(f"[{ts_str}] listen" if is_listen else f"[{ts_str}] no finding")
                    last_reported_sec = current_sec

            processed += 1

        # Flush any remaining speech at end of video
        if speech_buffer:
            flush_speech_buffer()

    if findings == 0:
        print("未检测到相关店铺。")
    else:
        print(f"Done. Processed {processed} sampled frames; findings: {findings}; saved to {output_path}")


def run_duplex_fps_benchmark(
    model: OVMiniCPMO,
    video_path: str,
    prompt: str,
    fps: float,
    limit_frames: int,
    warmup_frames: int,
    max_slice_nums: int,
    max_new_tokens: int,
) -> None:
    if warmup_frames < 0:
        raise ValueError("--benchmark-warmup must be non-negative")
    measured_frames = limit_frames or 12
    frames = list(iter_video_frames(video_path, fps))[: warmup_frames + measured_frames]
    if len(frames) <= warmup_frames:
        raise ValueError("Video does not contain enough sampled frames for the benchmark")

    duplex = model.as_duplex(generate_audio=False, force_listen_count=0, listen_prob_scale=0.2)
    duplex.prepare(
        prefix_system_prompt=prompt,
        generate_audio=False,
        force_listen_count=0,
        sliding_window_mode="context",
    )

    def process_frame(frame: Image.Image) -> tuple[float, float]:
        started = time.perf_counter()
        prefill = duplex.streaming_prefill(
            frame_list=[frame],
            text_list=None,
            max_slice_nums=max_slice_nums,
            batch_vision_feed=True,
        )
        prefill_elapsed = time.perf_counter() - started
        if not prefill.get("success"):
            raise RuntimeError(f"streaming_prefill failed: {prefill}")
        duplex.streaming_generate(
            max_new_speak_tokens_per_chunk=max_new_tokens,
            decode_mode="greedy",
            listen_prob_scale=0.2,
            text_repetition_penalty=1.05,
        )
        return prefill_elapsed, time.perf_counter() - started

    print(
        f"Duplex FPS benchmark: warmup={warmup_frames}; measured={len(frames) - warmup_frames}; "
        f"input_fps={fps}; max_slice_nums={max_slice_nums}; max_new_tokens={max_new_tokens}"
    )
    for _, _, frame in frames[:warmup_frames]:
        process_frame(frame)

    prefill_times: list[float] = []
    end_to_end_times: list[float] = []
    for index, (_, _, frame) in enumerate(frames[warmup_frames:], start=1):
        prefill_elapsed, total_elapsed = process_frame(frame)
        prefill_times.append(prefill_elapsed)
        end_to_end_times.append(total_elapsed)
        print(
            f"  frame {index:02d}: prefill={prefill_elapsed:.3f}s "
            f"end-to-end={total_elapsed:.3f}s"
        )

    def percentile_95(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]

    count = len(end_to_end_times)
    prefill_fps = count / sum(prefill_times)
    end_to_end_fps = count / sum(end_to_end_times)
    print(
        "Benchmark result:\n"
        f"  vision prefill throughput: {prefill_fps:.3f} FPS "
        f"(median {statistics.median(prefill_times):.3f}s, p95 {percentile_95(prefill_times):.3f}s)\n"
        f"  duplex end-to-end throughput: {end_to_end_fps:.3f} FPS "
        f"(median {statistics.median(end_to_end_times):.3f}s, p95 {percentile_95(end_to_end_times):.3f}s)\n"
        f"  sustainable frame interval: {1.0 / end_to_end_fps:.3f}s/frame"
    )


def main() -> None:
    args = parse_args()
    needs_model = args.smoke_test or not args.video or args.analysis_mode != "realtime" or args.vlm_every > 0
    model = load_model(args.model_path, args.device, args.tts_device) if needs_model else None

    if args.smoke_test or not args.video:
        if model is None:
            raise RuntimeError("Smoke test requires a loaded model")
        run_smoke_test(model, args.image, args.max_slice_nums, args.max_new_tokens)

    if args.video:
        if args.analysis_mode == "benchmark":
            if model is None:
                raise RuntimeError("Benchmark mode requires a loaded model")
            run_duplex_fps_benchmark(
                model=model,
                video_path=args.video,
                prompt=args.prompt,
                fps=args.fps,
                limit_frames=args.limit_frames,
                warmup_frames=args.benchmark_warmup,
                max_slice_nums=args.max_slice_nums,
                max_new_tokens=args.max_new_tokens,
            )
        elif args.analysis_mode == "hybrid":
            if model is None:
                raise RuntimeError("Hybrid mode requires a loaded primary model")
            print("Loading independent high-resolution refinement model...")
            refine_model = load_model(args.model_path, args.refine_device, args.tts_device)
            run_hybrid_video_scan(
                primary_model=model,
                refine_model=refine_model,
                video_path=args.video,
                fps=args.fps,
                limit_frames=args.limit_frames,
                window_frames=args.window_frames,
                window_stride=args.window_stride,
                max_slice_nums=args.max_slice_nums,
                max_new_tokens=args.max_new_tokens,
                refine_slice_nums=args.refine_slice_nums,
                refine_max_new_tokens=args.refine_max_new_tokens,
                refine_every=args.refine_every,
                output_jsonl=args.output_jsonl,
            )
        elif args.analysis_mode == "realtime":
            run_realtime_video_scan(
                model=model,
                video_path=args.video,
                fps=args.fps,
                limit_frames=args.limit_frames,
                window_frames=args.window_frames,
                window_stride=args.window_stride,
                max_slice_nums=args.max_slice_nums,
                max_new_tokens=args.max_new_tokens,
                output_jsonl=args.output_jsonl,
                enable_ocr=not args.disable_ocr,
                vlm_every=args.vlm_every,
            )
        elif args.analysis_mode == "calibrated":
            if model is None:
                raise RuntimeError("Calibrated mode requires a loaded model")
            run_calibrated_video_scan(
                model=model,
                video_path=args.video,
                fps=args.fps,
                limit_frames=args.limit_frames,
                window_frames=args.window_frames,
                window_stride=args.window_stride,
                max_slice_nums=args.max_slice_nums,
                max_new_tokens=args.max_new_tokens,
                output_jsonl=args.output_jsonl,
            )
        else:
            if model is None:
                raise RuntimeError("Duplex mode requires a loaded model")
            run_video_stream(
                model=model,
                video_path=args.video,
                prompt=args.prompt,
                fps=args.fps,
                limit_frames=args.limit_frames,
                max_slice_nums=args.max_slice_nums,
                max_new_tokens=args.max_new_tokens,
                output_jsonl=args.output_jsonl,
            )


if __name__ == "__main__":
    main()
