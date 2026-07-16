#!/usr/bin/env python3
"""Benchmark MiniCPM-o 4.5 on three-frame dashcam windows."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from duplex_baozi_video_demo import (
    REALTIME_SCAN_PROMPT,
    build_storefront_grid,
    canonicalize_finding,
    parse_json_array,
)
from minicpm_o_4_5_helper import OVMiniCPMO


EXPECTED_TARGETS = {"1点点", "确幸の茶", "南京灌汤小笼包"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--input-mode", choices=("grid", "independent"), default="grid")
    parser.add_argument("--sample-fps", type=float, default=6.0)
    parser.add_argument("--max-side", type=int, default=0,
                        help="Resize each independent frame to this maximum side; 0 keeps original")
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--warmup-windows", type=int, default=1)
    parser.add_argument("--limit-windows", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def iter_windows(video: Path, sample_fps: float):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(native_fps / sample_fps))
    frames: list[tuple[int, float, Image.Image]] = []
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if index % interval == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append((index, index / native_fps, Image.fromarray(rgb)))
                if len(frames) == 3:
                    yield frames
                    frames = []
            index += 1
    finally:
        capture.release()
    if frames:
        while len(frames) < 3:
            frames.append(frames[-1])
        yield frames


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0 or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def used_memory_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    return (values["MemTotal"] - values["MemAvailable"]) / 1024 / 1024


def content_for_window(
    frames: list[tuple[int, float, Image.Image]], input_mode: str, max_side: int
) -> tuple[list[Any], list[int]]:
    if input_mode == "grid":
        grid = build_storefront_grid(frames)
        return [grid, REALTIME_SCAN_PROMPT], list(grid.size)
    images = [resize_max_side(frame[2], max_side) for frame in frames]
    return [*images, REALTIME_SCAN_PROMPT], list(images[0].size)


def infer(
    model: OVMiniCPMO,
    content: list[Any],
    max_slice_nums: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    stream = model.chat(
        msgs=[{"role": "user", "content": content}],
        max_slice_nums=max_slice_nums,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        enable_thinking=False,
        generate_audio=False,
        repetition_penalty=1.08,
        stream=True,
    )
    first_output_at: float | None = None
    chunks: list[str] = []
    for chunk in stream:
        text = str(chunk or "")
        if text and first_output_at is None:
            first_output_at = time.perf_counter()
        chunks.append(text)
    finished = time.perf_counter()
    answer = "".join(chunks)
    if first_output_at is None:
        first_output_at = finished
    token_ids = model.tokenizer.encode(answer, add_special_tokens=False)
    output_tokens = len(token_ids)
    ttft = first_output_at - started
    e2e = finished - started
    decode_time = max(0.0, e2e - ttft)
    tps = ((output_tokens - 1) / decode_time) if output_tokens > 1 and decode_time > 0 else 0.0
    return {
        "answer": answer,
        "output_tokens": output_tokens,
        "ttft_s": ttft,
        "tps": tps,
        "e2e_s": e2e,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps must be positive")
    if args.warmup_windows < 0:
        raise SystemExit("--warmup-windows must not be negative")

    baseline_memory = used_memory_gib()
    peak_memory = baseline_memory
    stop_sampler = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_memory
        while not stop_sampler.wait(0.1):
            peak_memory = max(peak_memory, used_memory_gib())

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    rows: list[dict[str, Any]] = []
    found_targets: set[str] = set()
    try:
        model = OVMiniCPMO(model_path=args.model_path, device=args.device, tts_device="CPU")
        windows = iter_windows(args.video, args.sample_fps)
        for warmup_index in range(args.warmup_windows):
            try:
                frames = next(windows)
            except StopIteration as exc:
                raise RuntimeError("Video has too few windows for warmup") from exc
            content, size = content_for_window(frames, args.input_mode, args.max_side)
            print(f"warmup {warmup_index + 1}/{args.warmup_windows} input_size={size}", flush=True)
            infer(model, content, args.max_slice_nums, args.max_new_tokens)

        for index, frames in enumerate(windows, 1):
            if args.limit_windows and index > args.limit_windows:
                break
            content, size = content_for_window(frames, args.input_mode, args.max_side)
            measured = infer(model, content, args.max_slice_nums, args.max_new_tokens)
            recognized: list[str] = []
            for item in parse_json_array(measured["answer"]):
                normalized = canonicalize_finding(item)
                if normalized is None:
                    continue
                name = normalized[0]
                recognized.append(name)
                if name in EXPECTED_TARGETS:
                    found_targets.add(name)
            row = {
                "window": index,
                "frame_times_s": [round(frame[1], 6) for frame in frames],
                "input_size": size,
                "ttft_s": round(measured["ttft_s"], 6),
                "tps": round(measured["tps"], 6),
                "e2e_s": round(measured["e2e_s"], 6),
                "output_tokens": measured["output_tokens"],
                "recognized": recognized,
                "raw": measured["answer"],
            }
            rows.append(row)
            print(
                f"window={index:02d} size={size} ttft={row['ttft_s']:.3f}s "
                f"tps={row['tps']:.2f} e2e={row['e2e_s']:.3f}s "
                f"tokens={row['output_tokens']} recognized={recognized}",
                flush=True,
            )
    finally:
        stop_sampler.set()
        sampler.join()

    if not rows:
        raise RuntimeError("No measured windows; reduce warmup or check the video")
    hits = sorted(found_targets)
    tps_rows = [row for row in rows if row["tps"] > 0]
    summary = {
        "model": "MiniCPM-o-4.5 Omni 9B",
        "model_path": args.model_path,
        "device": args.device,
        "input_mode": args.input_mode,
        "sample_fps": args.sample_fps,
        "max_side": args.max_side or "original",
        "max_slice_nums": args.max_slice_nums,
        "max_new_tokens": args.max_new_tokens,
        "warmup_windows": args.warmup_windows,
        "measured_windows": len(rows),
        "ttft_s_mean": statistics.mean(row["ttft_s"] for row in rows),
        "tps_mean": statistics.mean(row["tps"] for row in tps_rows) if tps_rows else None,
        "tps_measured_windows": len(tps_rows),
        "e2e_s_mean": statistics.mean(row["e2e_s"] for row in rows),
        "system_memory_baseline_gib": baseline_memory,
        "system_memory_peak_gib": peak_memory,
        "system_memory_increment_gib": max(0.0, peak_memory - baseline_memory),
        "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
        "accuracy": f"{len(hits)}/{len(EXPECTED_TARGETS)}",
        "hits": hits,
        "misses": sorted(EXPECTED_TARGETS - found_targets),
        "rows": rows,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("SUMMARY=" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run(parse_args())
