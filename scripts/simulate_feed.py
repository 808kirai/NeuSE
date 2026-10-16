"""
feeds generated thermal frames through the real pipeline so the dashboard has data
the frames are made up but every number on the dashboard comes from the pipeline

usage
    python scripts/simulate_feed.py --frames 60 --interval 1
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# put the project root on the path so src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import dual_spectral_pipeline


def synthetic_thermal_frame(step, rng, height=480, width=640, targets=2):
    # cool noisy background like rubble at night
    frame = rng.integers(0, 80, (height, width), dtype=np.uint8)
    for i in range(targets):
        # each warm body drifts slowly along its own path
        cx = int((150 + 220 * i + 6 * step) % (width - 60)) + 30
        cy = int(140 + 120 * i + 40 * np.sin(step / 8 + i))
        cv2.ellipse(frame, (cx, cy), (18, 30), 0, 0, 360, 245, -1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="feed generated thermal frames through the pipeline")
    parser.add_argument("--frames", type=int, default=60, help="how many frames to process")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between frames")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the background noise")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pipe = dual_spectral_pipeline()

    for step in range(args.frames):
        hits = pipe.process_frames(thermal_frame=synthetic_thermal_frame(step, rng))
        print(f"frame {step + 1}/{args.frames}  {len(hits)} detections in {pipe.last_latency_ms:.2f}ms")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
