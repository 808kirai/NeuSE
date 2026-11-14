"""
trains the person check used by the classical detector, on an ordinary cpu in a few minutes
learns from the hit-uav training images, then uses the validation images to
    calibrate the scores, so a score of 0.8 means about 8 in 10 such hits were people
    pick the score a hit needs to be reported
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
from src.metrics import add_counts, center_inside, iou, match_image, summarize
from src.person_check import (feature_count, fitted_boxes, frame_features, raw_scores, suppress_overlaps,
                              to_chance)

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
    rows = np.array(rows, np.float32).reshape(-1, feature_count)
    return rows, np.array(labels, np.float32), np.array(targets, np.float32).reshape(-1, 4), owners, boxes


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


def fit(rows, labels, targets, epochs=30):
    # learns the network and the box fitting from one set of candidates
    mean, std = rows.mean(0), rows.std(0) + 1e-6
    z = (rows - mean) / std
    w1, b1, w2, b2 = train_network(z, labels, epochs=epochs)
    return {"mean": mean, "std": std, "w1": w1, "b1": b1, "w2": w2, "b2": b2,
            "box_weights": fit_box_weights(z, targets, labels)}


def apply(model, rows, boxes):
    # the network's raw output and the fitted box for each candidate, with the same code the detector runs
    raw = raw_scores(rows, model["mean"], model["std"], model["w1"], model["b1"], model["w2"], model["b2"])
    return raw, fitted_boxes(rows, boxes, model["mean"], model["std"], model["box_weights"])


def by_image(records, values, fitted, owners, least=None):
    # groups candidates by image and drops overlapping boxes, keeping the best of each
    per_image = [[] for _ in records]
    for box, value, owner in zip(fitted, values, owners):
        if least is None or value >= least:
            per_image[owner].append((box, float(value)))
    return [suppress_overlaps(found, overlap) for found in per_image]


def calibrate(records, raw, fitted, owners):
    # fits scale and shift so the chance given to a hit matches how often such hits are people
    # each kept box counts as right when it overlaps a labelled person by at least half
    outputs, right = [], []
    for record, found in zip(records, by_image(records, raw, fitted, owners)):
        claimed = set()
        # best scores claim people first, the same way the scoring script counts hits
        for box, value in found:
            best, best_overlap = None, 0.5
            for p, person in enumerate(record["people"]):
                shared = iou(box, person)
                if p not in claimed and shared >= best_overlap:
                    best, best_overlap = p, shared
            if best is not None:
                claimed.add(best)
            elif any(center_inside(box, region) for region in record["ignore"]):
                # hits on dont care regions say nothing either way
                continue
            outputs.append(value)
            right.append(1.0 if best is not None else 0.0)

    x, y = np.array(outputs, np.float64), np.array(right, np.float64)
    if len(x) == 0 or y.min() == y.max():
        # nothing to fit against, so leave the scores as they are
        return 1.0, 0.0

    def loss(scale, shift):
        chance = np.clip(1 / (1 + np.exp(-np.clip(scale * x + shift, -30, 30))), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(chance) + (1 - y) * np.log(1 - chance))

    # newton's method on a two number logistic fit, taking smaller steps whenever one would make it worse
    scale, shift = 1.0, 0.0
    for _ in range(100):
        chance = 1 / (1 + np.exp(-np.clip(scale * x + shift, -30, 30)))
        gradient = np.array([np.mean((chance - y) * x), np.mean(chance - y)])
        weight = chance * (1 - chance)
        hessian = np.array([[np.mean(weight * x * x) + 1e-9, np.mean(weight * x)],
                            [np.mean(weight * x), np.mean(weight) + 1e-9]])
        step = np.linalg.solve(hessian, gradient)
        size, before = 1.0, loss(scale, shift)
        while size > 1e-6 and loss(scale - size * step[0], shift - size * step[1]) > before:
            size /= 2
        scale, shift = scale - size * step[0], shift - size * step[1]
        if np.abs(size * step).max() < 1e-9:
            break
    # a fit that turns the order upside down would be wrong, so fall back to the raw scores
    return (float(scale), float(shift)) if scale > 0 else (1.0, 0.0)


def pick_threshold(records, scores, fitted, owners):
    # tries many score cut offs on the validation images and keeps the one with the best f1
    best = (0.0, 0.5)
    for threshold in np.quantile(scores, np.linspace(0.5, 0.999, 120)):
        total = {}
        for record, found in zip(records, by_image(records, scores, fitted, owners, least=threshold)):
            add_counts(total, match_image([box for box, _ in found], record["people"], record["ignore"],
                                          record["others"]))
        f1 = summarize(total)["f1"]
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best


def tune(model, records, rows, owners, boxes):
    # uses held out images to calibrate the scores and pick the reporting threshold
    # returns the validation f1 with the settings added to the model
    raw, fitted = apply(model, rows, boxes)
    scale, shift = calibrate(records, raw, fitted, owners)
    f1, threshold = pick_threshold(records, to_chance(raw, scale, shift), fitted, owners)
    model.update(scale=np.float64(scale), shift=np.float64(shift), threshold=np.float64(threshold),
                 overlap=np.float32(overlap))
    return f1


def train(dataset, out, epochs=30):
    # trains, calibrates, picks the threshold and saves the weights, returns the validation f1 and threshold
    start = time.time()
    train_rows, train_labels, train_targets, _, _ = collect(load_split(dataset, "train"))
    val_records = load_split(dataset, "val")
    val_rows, _, _, val_owners, val_boxes = collect(val_records)
    print(f"{len(train_rows)} training spots, {int(train_labels.sum())} on people, in {time.time() - start:.0f} s")

    model = fit(train_rows, train_labels, train_targets, epochs)
    f1 = tune(model, val_records, val_rows, val_owners, val_boxes)
    print(f"validation f1 {f1:.3f} at threshold {float(model['threshold']):.3f}, "
          f"calibration scale {float(model['scale']):.3f} shift {float(model['shift']):.3f}")

    np.savez(out, **model)
    print(f"saved {out} after {time.time() - start:.0f} s")
    return f1, float(model["threshold"])


def main():
    parser = argparse.ArgumentParser(description="train the classical person check on hit-uav")
    parser.add_argument("--dataset", required=True, help="folder of the cloned hit-uav repository")
    parser.add_argument("--out", default=str(default_out), help="where to save the weights")
    args = parser.parse_args()
    train(args.dataset, args.out)


if __name__ == "__main__":
    main()
