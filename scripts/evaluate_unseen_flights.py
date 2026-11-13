"""
scores the person check on flights it has never seen
hit-uav's own train, validation and test images come from the same flights, often a few frames apart
so its test score shows how the detector does on those scenes, not on a new place

here the images are grouped by flight setting, the time of day, height and camera angle in the file name
the groups are dealt into 5 parts, and each part in turn is scored by a detector trained without it
the next part along is used to calibrate and pick the threshold, and the other three to train

this trains 5 detectors, each on about three fifths of the images, and takes about 20 minutes on a laptop
the saved weights in models/ are not touched

usage
    python scripts/evaluate_unseen_flights.py --dataset C:/Users/you/datasets/hit-uav
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# put the project root on the path so src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.hit_uav import load_split, splits
from src.metrics import add_counts, average_precision, center_inside, match_image, summarize
from src.person_check import feature_count, frame_features, to_chance
from train_person_check import apply, by_image, fit, tune

parts = 5


def flight(record):
    # images with the same time of day, height, camera angle and weather count as one flight setting
    return "_".join(Path(record["file"]).stem.split("_")[:4])


def deal(records):
    # deals whole flight settings into parts of about the same number of images
    # the biggest settings are placed first, each into the part that is smallest so far
    sizes = {}
    for record in records:
        sizes[flight(record)] = sizes.get(flight(record), 0) + 1
    part_of, totals = {}, [0] * parts
    for name in sorted(sizes, key=lambda n: (-sizes[n], n)):
        smallest = totals.index(min(totals))
        part_of[name] = smallest
        totals[smallest] += sizes[name]
    return part_of


def candidates(records):
    # features, labels and box targets for every candidate in every image, worked out once
    found = []
    for record in records:
        gray = cv2.imread(str(record["path"]), cv2.IMREAD_GRAYSCALE)
        rows, boxes = frame_features(gray)
        labels, targets = [], []
        for box in boxes:
            person = next((p for p in record["people"] if center_inside(box, p)), None)
            labels.append(1.0 if person else 0.0)
            if person:
                x, y, w, h = box
                targets.append([person[0] + person[2] / 2 - (x + w / 2), person[1] + person[3] / 2 - (y + h / 2),
                                np.log(person[2]), np.log(person[3])])
            else:
                targets.append([0, 0, 0, 0])
        found.append((rows, boxes, np.array(labels, np.float32), np.array(targets, np.float32).reshape(-1, 4)))
    return found


def stack(found, indexes):
    # joins the candidates of the chosen images, owners says which image each one came from
    rows = np.concatenate([found[i][0] for i in indexes]) if indexes else np.zeros((0, feature_count), np.float32)
    labels = np.concatenate([found[i][2] for i in indexes])
    targets = np.concatenate([found[i][3] for i in indexes])
    boxes = [box for i in indexes for box in found[i][1]]
    owners = [n for n, i in enumerate(indexes) for _ in found[i][1]]
    return rows, labels, targets, boxes, owners


def main():
    parser = argparse.ArgumentParser(description="score the person check on flight settings held out of training")
    parser.add_argument("--dataset", required=True, help="folder of the cloned hit-uav repository")
    parser.add_argument("--json-out", help="also save the results to this file")
    args = parser.parse_args()

    start = time.time()
    records = [r for split in splits for r in load_split(args.dataset, split)]
    part_of = deal(records)
    found = candidates(records)
    print(f"{len(records)} images in {len(part_of)} flight settings, features in {time.time() - start:.0f} s")

    results, total = [], {}
    for held_out in range(parts):
        tuning = (held_out + 1) % parts
        chosen = {"train": [], "tune": [], "test": []}
        for i, record in enumerate(records):
            part = part_of[flight(record)]
            chosen["test" if part == held_out else "tune" if part == tuning else "train"].append(i)

        rows, labels, targets, _, _ = stack(found, chosen["train"])
        model = fit(rows, labels, targets)

        # calibrate and pick the threshold on flights that were not trained on either
        tune_records = [records[i] for i in chosen["tune"]]
        rows, _, _, boxes, owners = stack(found, chosen["tune"])
        tune(model, tune_records, rows, owners, boxes)

        # score the held out flights
        test_records = [records[i] for i in chosen["test"]]
        rows, _, _, boxes, owners = stack(found, chosen["test"])
        raw, fitted = apply(model, rows, boxes)
        scores = to_chance(raw, float(model["scale"]), float(model["shift"]))
        ranked = by_image(test_records, scores, fitted, owners)
        ap = average_precision([(dets, r["people"], r["ignore"]) for r, dets in zip(test_records, ranked)])

        counts = {}
        threshold = float(model["threshold"])
        for record, dets in zip(test_records, ranked):
            kept = [box for box, score in dets if score >= threshold]
            add_counts(counts, match_image(kept, record["people"], record["ignore"], record["others"]))
        add_counts(total, counts)
        m = summarize(counts)
        results.append({"part": held_out + 1, "images": len(test_records),
                        "people": sum(len(r["people"]) for r in test_records), "map50": ap,
                        "precision": m["precision"], "recall": m["recall"],
                        "false_per_image": m["fp"] / len(test_records)})
        print(f"part {held_out + 1}: {len(test_records)} images, {results[-1]['people']} people, "
              f"map@0.5 {ap:.3f}, precision {m['precision']:.3f}, recall {m['recall']:.3f}, "
              f"{results[-1]['false_per_image']:.2f} false hits per image, {time.time() - start:.0f} s")

    overall = summarize(total)
    aps = [r["map50"] for r in results]
    print(f"\nunseen flights, mean map@0.5 {np.mean(aps):.3f}, lowest part {min(aps):.3f}, highest {max(aps):.3f}")
    print(f"all parts together, precision {overall['precision']:.3f}, recall {overall['recall']:.3f}, "
          f"{overall['fp'] / len(records):.2f} false hits per image")

    if args.json_out:
        summary = {"parts": results, "mean_map50": float(np.mean(aps)), "precision": overall["precision"],
                   "recall": overall["recall"], "false_per_image": overall["fp"] / len(records)}
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
