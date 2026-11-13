"""
scores the classical baseline against the real person labels in hit-uav
nothing is written to the detection database

two ways of counting a hit
    iou, the box must overlap a person box by at least half, the usual standard for detectors
    center, the centre of the box only has to land on a person, looser and kinder to a blob detector

map@0.5 is also reported, the main number yolo tools give
the baseline has no confidence, so its hits are ranked by their brightest pixel instead

usage
    python scripts/evaluate_hit_uav.py --dataset C:/Users/you/datasets/hit-uav --split test
    python scripts/evaluate_hit_uav.py --dataset C:/Users/you/datasets/hit-uav --split val --sweep
    python scripts/evaluate_hit_uav.py --dataset ... --split test --threshold 200 --min-area 60 --save-vis C:/Users/you/neuse-vis
"""
import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

# put the project root on the path so src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.hit_uav import load_split
from src.metrics import add_counts, average_precision, match_image, summarize
from src.pipeline import dual_spectral_pipeline

rules = ("iou", "center")
groups = ("all", "day", "night", "60-80 m", "90-110 m", "120-130 m")

# config settings this script may change, all put back after each run
tunable = ("thermal_threshold", "min_blob_area", "clahe_clip_limit", "clahe_tile_grid")

# settings tried by --sweep
sweep_thresholds = (150, 170, 190, 200, 210, 230)
sweep_min_areas = (5, 10, 20, 40, 60, 100)


def altitude_band(altitude):
    # groups the eight flight heights into low, middle and high
    if altitude <= 80:
        return "60-80 m"
    if altitude <= 110:
        return "90-110 m"
    return "120-130 m"


def baseline_detector():
    # returns a function that takes an 8 bit frame and gives back (box, score) pairs
    # a trained model plugs in here later with the same shape, so the scoring stays the same
    pipe = dual_spectral_pipeline()

    def detect(frame):
        # calls the detector directly so the database is left alone
        hits = pipe._detect(frame)
        # no confidence from the baseline, so rank by brightness for map
        return [(hit["bbox"], hit["confidence"] if hit["confidence"] is not None else hit["thermal_peak"] / 255)
                for hit in hits]

    return detect


def detect_all(records, detect):
    # runs the detector on every image and returns the hits and time per image
    detections, times = [], []
    for record in records:
        frame = cv2.imread(str(record["path"]), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise FileNotFoundError(f"could not read {record['path']}")
        start = time.perf_counter()
        detections.append(detect(frame))
        times.append((time.perf_counter() - start) * 1000)
    return detections, times


def in_group(record, group):
    return group in ("all", record["time"], altitude_band(record["altitude"]))


def score(records, detections, rule):
    # precision and recall for the whole split plus by time of day and altitude
    totals = {group: {} for group in groups}
    for record, dets in zip(records, detections):
        counts = match_image([box for box, _ in dets], record["people"], record["ignore"], record["others"], rule=rule)
        for group in groups:
            if in_group(record, group):
                add_counts(totals[group], counts)
    # groups with no images are left out
    return {group: summarize(total) for group, total in totals.items() if total}


def map50(records, detections):
    # map@0.5 for the whole split plus each group
    result = {}
    for group in groups:
        images = [(dets, r["people"], r["ignore"]) for r, dets in zip(records, detections) if in_group(r, group)]
        if images:
            result[group] = average_precision(images)
    return result


def save_vis(records, detections, out_dir, count):
    # saves the worst frames side by side, raw on the left and boxes on the right
    # green are labelled people, yellow are vehicles, grey are dont care regions and red are detector hits
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = []
    for record, dets in zip(records, detections):
        counts = match_image([box for box, _ in dets], record["people"], record["ignore"], record["others"])
        scored.append((counts["fp"] + counts["fn"], counts, record, dets))
    # most mistakes first, file name breaks ties so the pick is the same every run
    scored.sort(key=lambda item: (-item[0], item[2]["file"]))

    saved = []
    for rank, (_, counts, record, dets) in enumerate(scored[:count], start=1):
        raw = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
        marked = raw.copy()
        for colour, boxes in (((160, 160, 160), record["ignore"]), ((0, 220, 255), record["others"]),
                              ((0, 200, 0), record["people"]), ((0, 0, 255), [box for box, _ in dets])):
            for x, y, w, h in boxes:
                cv2.rectangle(marked, (int(x), int(y)), (int(x + w), int(y + h)), colour, 1)
        note = f"{record['time']} {record['altitude']} m  found {counts['tp']}  missed {counts['fn']}  false {counts['fp']}"
        cv2.putText(marked, note, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        path = out_dir / f"{rank:02d}_{record['file']}"
        cv2.imwrite(str(path), cv2.hconcat([raw, marked]))
        saved.append(path)
    return saved


def evaluate(records, vis_dir=None, vis_count=20, **settings):
    # scores one setting, e.g. evaluate(records, thermal_threshold=200, min_blob_area=60)
    # unset settings keep their config value, and config is put back afterwards
    unknown = set(settings) - set(tunable)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}")
    saved = {name: getattr(config, name) for name in tunable}
    try:
        for name, value in settings.items():
            if value is not None:
                setattr(config, name, value)
        used = {name: getattr(config, name) for name in tunable}
        # the detector reads config when built, so build it after the changes
        detections, times = detect_all(records, baseline_detector())
    finally:
        for name, value in saved.items():
            setattr(config, name, value)

    if vis_dir:
        save_vis(records, detections, vis_dir, vis_count)

    return {
        "settings": used,
        "images": len(records),
        "people": sum(len(r["people"]) for r in records),
        "median_ms": statistics.median(times),
        "map50": map50(records, detections),
        "scores": {rule: score(records, detections, rule) for rule in rules},
    }


def print_report(result):
    print(", ".join(f"{name} {value}" for name, value in result["settings"].items()))
    print(f"{result['images']} images, {result['people']} people, median {result['median_ms']:.1f} ms per image")
    print("\nmap@0.5, hits ranked by brightness")
    for group, value in result["map50"].items():
        print(f"  {group:10} {value:.3f}")
    for rule in rules:
        print(f"\nhit rule {rule}")
        for group, m in result["scores"][rule].items():
            print(f"  {group:10} precision {m['precision']:.3f}  recall {m['recall']:.3f}  f1 {m['f1']:.3f}"
                  f"  found {m['tp']}  missed {m['fn']}  false {m['fp']}  false on vehicles {m['fp_on_other']}")


def main():
    parser = argparse.ArgumentParser(description="score the classical baseline on hit-uav person labels")
    parser.add_argument("--dataset", required=True, help="folder of the cloned hit-uav repository")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--threshold", type=int, help="override thermal_threshold")
    parser.add_argument("--min-area", type=int, help="override min_blob_area")
    parser.add_argument("--clip-limit", type=float, help="override clahe_clip_limit")
    parser.add_argument("--tile-grid", type=int, help="override clahe_tile_grid")
    parser.add_argument("--sweep", action="store_true", help="try a grid of threshold and area settings, ranked by f1")
    parser.add_argument("--save-vis", help="folder to save the worst frames with their boxes drawn")
    parser.add_argument("--vis-count", type=int, default=20, help="how many frames --save-vis keeps")
    parser.add_argument("--json-out", help="also save the full results to this file")
    args = parser.parse_args()

    records = load_split(args.dataset, args.split)
    fixed = {"clahe_clip_limit": args.clip_limit, "clahe_tile_grid": args.tile_grid}

    if args.sweep:
        # pick settings on val, then check them once on test so the test number stays fair
        results = []
        for threshold, min_area in itertools.product(sweep_thresholds, sweep_min_areas):
            result = evaluate(records, thermal_threshold=threshold, min_blob_area=min_area, **fixed)
            results.append(result)
            print(f"threshold {threshold:3}  min_area {min_area:3}  "
                  f"iou f1 {result['scores']['iou']['all']['f1']:.3f}  "
                  f"center f1 {result['scores']['center']['all']['f1']:.3f}  "
                  f"map@0.5 {result['map50']['all']:.3f}")
        best = max(results, key=lambda r: r["scores"]["iou"]["all"]["f1"])
        print("\nbest by iou f1")
        print_report(best)
    else:
        results = evaluate(records, args.save_vis, args.vis_count, thermal_threshold=args.threshold,
                           min_blob_area=args.min_area, **fixed)
        print_report(results)
        if args.save_vis:
            print(f"\nsaved the {args.vis_count} worst frames to {args.save_vis}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
