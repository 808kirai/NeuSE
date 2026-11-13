"""
feeds thermal frames through the real pipeline so the dashboard has data
every number on the dashboard comes from the pipeline

made up frames have large warm shapes, so they go through the warm-spots detector
--replay sends real drone frames, like the hit-uav test images, through the person check instead

usage
    python scripts/simulate_feed.py --frames 60 --interval 1
    python scripts/simulate_feed.py --raw16 --dead-pixels 3
    python scripts/simulate_feed.py --replay C:/Users/you/datasets/hit-uav/normal_json/test --frames 30
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# put the project root on the path so src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import thermal_pipeline


def synthetic_thermal_frame(step, rng, height=480, width=640, targets=2):
    # builds one fake thermal frame
    # step moves the warm bodies a little so each frame is different

    # cool noisy background like rubble at night
    frame = rng.integers(0, 80, (height, width), dtype=np.uint8)
    for i in range(targets):
        # each warm body drifts slowly along its own path
        cx = int((150 + 220 * i + 6 * step) % (width - 60)) + 30
        cy = int(140 + 120 * i + 40 * np.sin(step / 8 + i))
        # draw a warm oval roughly the size of a person seen from above
        cv2.ellipse(frame, (cx, cy), (18, 30), 0, 0, 360, 245, -1)
    return frame


def to_raw16(frame, rng, dead_pixels=0):
    # turns an 8 bit frame into 16 bit counts like a radiometric camera gives
    # background lands near 7000 and warm bodies near 9000
    raw = (7000 + frame.astype(np.uint16) * 8).astype(np.uint16)
    # stuck hot pixels like real sensors have
    for _ in range(dead_pixels):
        raw[rng.integers(0, raw.shape[0]), rng.integers(0, raw.shape[1])] = 65535
    return raw


def replay(args):
    # sends real thermal images through the configured detector in name order
    images = sorted(Path(args.replay).glob("*.jpg")) + sorted(Path(args.replay).glob("*.png"))
    if not images:
        sys.exit(f"no jpg or png images in {args.replay}")
    pipe = thermal_pipeline()
    for step, path in enumerate(images[:args.frames]):
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        hits = pipe.process_frames(thermal_frame=frame)
        print(f"{path.name}  {len(hits)} detections in {pipe.last_latency_ms:.2f}ms")
        time.sleep(args.interval)


def main():
    # read the command line options
    parser = argparse.ArgumentParser(description="feed generated thermal frames through the pipeline")
    parser.add_argument("--frames", type=int, default=60, help="how many frames to process")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between frames")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the background noise")
    parser.add_argument("--raw16", action="store_true", help="send 16 bit frames like a radiometric camera")
    parser.add_argument("--dead-pixels", type=int, default=0, help="stuck hot pixels per frame, needs --raw16")
    parser.add_argument("--replay", help="folder of real thermal images to send instead of made up frames")
    args = parser.parse_args()

    if args.replay:
        replay(args)
        return

    # a fixed seed makes every run produce the same frames
    rng = np.random.default_rng(args.seed)
    pipe = thermal_pipeline("warm-spots")

    for step in range(args.frames):
        frame = synthetic_thermal_frame(step, rng)
        if args.raw16:
            frame = to_raw16(frame, rng, args.dead_pixels)
        # only a thermal frame is sent since the baseline ignores rgb
        hits = pipe.process_frames(thermal_frame=frame)
        print(f"frame {step + 1}/{args.frames}  {len(hits)} detections in {pipe.last_latency_ms:.2f}ms")
        # wait so the dashboard updates at a readable pace
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
