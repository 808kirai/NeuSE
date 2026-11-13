"""
trains the person check used by the classical detector, on an ordinary cpu in a few minutes
learns from the hit-uav training images, then picks the score threshold on the validation images
the test images are never touched here, so the test score stays fair

usage
    python scripts/train_person_check.py --dataset C:/Users/you/datasets/hit-uav
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# put the project root on the path so src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hit_uav import load_split
from src.metrics import add_counts, center_inside, match_image, summarize
from src.person_check import frame_features, suppress_overlaps

default_out = Path(__file__).resolve().parent.parent / "models" / "person_check.npz"

# boxes overlapping more than this are treated as the same person
overlap = 0.5


def collect(records):
    # features for every candidate, whether it sits on a person, and where the person box is
    rows, labels, targets, owners, boxes = [], [], [], [], []
    for index, record in enumerate(records):
        gray = cv2.imread(str(record["path"]), cv2.IMREAD_GRAYSCALE)
        features, spots = frame_features(gray)
        for row, box in zip(features, spots):
            person = next((p for p in record["people"] if center_inside(box, p)), None)
            rows.append(row)
            labels.append(1.0 if person else 0.0)
            # how far the person centre is from the spot centre, and the person size as logs
            if person:
                x, y, w, h = box
                targets.append([person[0] + person[2] / 2 - (x + w / 2), person[1] + person[3] / 2 - (y + h / 2),
                                np.log(person[2]), np.log(person[3])])
            else:
                targets.append([0, 0, 0, 0])
            owners.append(index)
            boxes.append(box)
    return np.array(rows, np.float32), np.array(labels, np.float32), np.array(targets, np.float32), owners, boxes


def train_network(z, labels, hidden=128, epochs=30, rate=1e-3, decay=1e-4, seed=0):
    # one hidden layer trained with adam, people are rare so they get extra weight
    rng = np.random.default_rng(seed)
    count, width = z.shape
    w1 = rng.normal(0, np.sqrt(2 / width), (width, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, np.float32)
    w2 = rng.normal(0, np.sqrt(1 / hidden), (hidden, 1)).astype(np.float32)
    b2 = np.zeros(1, np.float32)
    params = [w1, b1, w2, b2]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    share = labels.mean()
    person_weight = np.sqrt((1 - share) / share) if 0 < share < 1 else 1.0

    step = 0
    for _ in range(epochs):
        order = rng.permutation(count)
        for start in range(0, count, 1024):
            batch = order[start:start + 1024]
            x, y = z[batch], labels[batch][:, None]
            hid = np.maximum(x @ w1 + b1, 0)
            chance = 1 / (1 + np.exp(-np.clip(hid @ w2 + b2, -30, 30)))
            error = (chance - y) * np.where(y == 1, person_weight, 1.0) / len(batch)
            back = (error @ w2.T) * (hid > 0)
            grads = [x.T @ back + decay * w1, back.sum(0), hid.T @ error + decay * w2, error.sum(0)]

            step += 1
            for i, (param, grad) in enumerate(zip(params, grads)):
                first[i] = 0.9 * first[i] + 0.1 * grad
                second[i] = 0.999 * second[i] + 0.001 * grad * grad
                param -= rate * (first[i] / (1 - 0.9 ** step)) / (np.sqrt(second[i] / (1 - 0.999 ** step)) + 1e-8)
    return w1, b1, w2, b2


def fit_box_weights(z, targets, labels, strength=10.0):
    # ridge regression from features to box shift and size, learned on spots that sit on people
    on_person = labels == 1
    a = np.hstack([z[on_person], np.ones((on_person.sum(), 1), np.float32)])
    return np.linalg.solve(a.T @ a + strength * np.eye(a.shape[1]), a.T @ targets[on_person]).astype(np.float32)


def pick_threshold(records, scores, fitted, owners):
    # tries many score cut offs on the validation images and keeps the one with the best f1
    best = (0.0, 0.5)
    for threshold in np.quantile(scores, np.linspace(0.5, 0.999, 120)):
        per_image = [[] for _ in records]
        for box, score, owner in zip(fitted, scores, owners):
            if score >= threshold:
                per_image[owner].append((box, float(score)))
        total = {}
        for record, found in zip(records, per_image):
            kept = [box for box, _ in suppress_overlaps(found, overlap)]
            add_counts(total, match_image(kept, record["people"], record["ignore"], record["others"]))
        f1 = summarize(total)["f1"]
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best


def train(dataset, out, epochs=30):
    # trains, picks the threshold and saves the weights, returns the validation f1 and threshold
    start = time.time()
    train_rows, train_labels, train_targets, _, _ = collect(load_split(dataset, "train"))
    val_records = load_split(dataset, "val")
    val_rows, _, _, val_owners, val_boxes = collect(val_records)
    print(f"{len(train_rows)} training spots, {int(train_labels.sum())} on people, in {time.time() - start:.0f} s")

    mean, std = train_rows.mean(0), train_rows.std(0) + 1e-6
    z = (train_rows - mean) / std
    w1, b1, w2, b2 = train_network(z, train_labels, epochs=epochs)
    box_weights = fit_box_weights(z, train_targets, train_labels)

    # score the validation spots with the new weights to pick the threshold
    val_z = (val_rows - mean) / std
    scores = (1 / (1 + np.exp(-np.clip(np.maximum(val_z @ w1 + b1, 0) @ w2 + b2, -30, 30)))).ravel()
    shift = np.hstack([val_z, np.ones((len(val_z), 1), np.float32)]) @ box_weights
    fitted = []
    for (x, y, w, h), (dx, dy, log_w, log_h) in zip(val_boxes, shift):
        new_w, new_h = float(np.exp(log_w)), float(np.exp(log_h))
        fitted.append([x + w / 2 + dx - new_w / 2, y + h / 2 + dy - new_h / 2, new_w, new_h])
    f1, threshold = pick_threshold(val_records, scores, fitted, val_owners)
    print(f"validation f1 {f1:.3f} at threshold {threshold:.3f}")

    np.savez(out, mean=mean, std=std, w1=w1, b1=b1, w2=w2, b2=b2, box_weights=box_weights,
             threshold=np.float32(threshold), overlap=np.float32(overlap))
    print(f"saved {out} after {time.time() - start:.0f} s")
    return f1, threshold


def main():
    parser = argparse.ArgumentParser(description="train the classical person check on hit-uav")
    parser.add_argument("--dataset", required=True, help="folder of the cloned hit-uav repository")
    parser.add_argument("--out", default=str(default_out), help="where to save the weights")
    args = parser.parse_args()
    train(args.dataset, args.out)


if __name__ == "__main__":
    main()
