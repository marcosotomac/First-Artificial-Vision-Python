#!/usr/bin/env python3

from __future__ import annotations


import argparse
import time
from pathlib import Path
from typing import Union

import cv2
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real‑time object detection with YOLOv8")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source. 0 for default webcam, 1 for external cam, or path/URL to video/stream",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("yolov8n.pt"),
        help="Path to YOLO model weights (*.pt). Download from https://github.com/ultralytics/ultralytics",  # noqa: E501
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (pixels); higher gives better accuracy at lower FPS",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for displaying detections [0.0 – 1.0]",
    )
    return parser


def open_source(src: str) -> cv2.VideoCapture:
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"❌ Could not open video source: {src}")
    return cap


def draw_fps(frame, fps: float) -> None:
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main(args: argparse.Namespace) -> None:
    model = YOLO(str(args.weights))
    cap = open_source(args.source)

    window_name = "YOLOv8 – press 'q' to quit"
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ End of stream or cannot read frame. Exiting…")
            break

        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

        annotated = results[0].plot()

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        draw_fps(annotated, fps)

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    cli_args.source = str(cli_args.source)
    main(cli_args)
