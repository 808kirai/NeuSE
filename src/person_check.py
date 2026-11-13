"""
classical person detector for drone thermal footage
step 1 finds small warm spots by comparing each pixel with its surroundings, so sunlit ground stays dark
step 2 scores each spot as person or not with a tiny neural network run in plain numpy
step 3 resizes each box to fit a person and drops overlapping boxes

the network and box sizes are learned by scripts/train_person_check.py from the hit-uav training images
everything runs on an ordinary cpu with only numpy and opencv
"""
from pathlib import Path

import cv2
import numpy as np

# a pixel is a candidate when it is this much warmer than the mean of the window around it
local_window = 15
local_contrast = 25

# warm spots outside this size in pixels are not person sized at 60 to 130 m
min_area = 15
max_area = 800

# each spot is described by the patch around it, split into cells of gradient directions
patch_size = 32
cell_size = 8
direction_bins = 9

cells = patch_size // cell_size
cell_index = (np.arange(patch_size)[:, None] // cell_size) * cells + np.arange(patch_size)[None, :] // cell_size
feature_count = cells * cells * direction_bins + 12 * 12 + 9 + 2


def shrink_weights(size=patch_size, small=12):
    # opencv's area shrink is a fixed weighted average of pixels
    # pushing each single pixel through it once gives those weights, so many patches shrink in one multiply
    weights = np.zeros((small * small, size * size), np.float32)
    for i in range(size * size):
        pixel = np.zeros(size * size, np.float32)
        pixel[i] = 1
        weights[:, i] = cv2.resize(pixel.reshape(size, size), (small, small), interpolation=cv2.INTER_AREA).ravel()
    return weights


shrink = shrink_weights()


def find_candidates(gray):
    # returns the warm spot stats and centres, plus the local contrast map
    image = gray.astype(np.float32)
    contrast = image - cv2.blur(image, (local_window, local_window))
    mask = (contrast > local_contrast).astype(np.uint8)
    _, _, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # row 0 is the background
    keep = [(s, c) for s, c in zip(stats[1:].tolist(), centres[1:].tolist()) if min_area <= s[4] <= max_area]
    return keep, contrast


def frame_features(gray, limit=None):
    # returns one feature row and one box per candidate in the frame
    # limit keeps only the warmest candidates, so a frame full of noise cannot stall the unit
    candidates, contrast = find_candidates(gray)
    if not candidates:
        return np.zeros((0, feature_count), np.float32), []
    if limit and len(candidates) > limit:
        peaks = [float(contrast[y:y + h, x:x + w].max()) for (x, y, w, h, _), _ in candidates]
        warmest = sorted(range(len(candidates)), key=lambda i: -peaks[i])[:limit]
        candidates = [candidates[i] for i in sorted(warmest)]

    # gradients once per frame, padded so patches at the edge stay full size
    padded = cv2.copyMakeBorder(gray, patch_size, patch_size, patch_size, patch_size,
                                cv2.BORDER_REFLECT).astype(np.float32)
    gx = cv2.Sobel(padded, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(padded, cv2.CV_32F, 0, 1, ksize=1)
    # edge strength and direction in one fast opencv call
    # its angles are within a fraction of a degree, and training uses this same code so they match
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # 9 bins across 0 to 180 degrees, an edge pointing the opposite way shares a bin
    # angles are never negative, so cutting off the fraction rounds down, much faster than numpy's floor division
    direction = (angle / (180 / direction_bins)).astype(np.int32)
    direction[direction >= direction_bins] -= direction_bins

    # patch corners for every candidate at once, in padded coordinates
    count = len(candidates)
    stats = np.array([s for s, _ in candidates], np.int64)
    centres = np.array([c for _, c in candidates])
    tops = np.round(centres[:, 1]).astype(np.int64) + patch_size // 2
    lefts = np.round(centres[:, 0]).astype(np.int64) + patch_size // 2
    grab = lambda image: np.lib.stride_tricks.sliding_window_view(image, (patch_size, patch_size))[tops, lefts]
    patches = grab(padded)

    # how strongly edges point each way in each cell, like hog
    bin_count = cells * cells * direction_bins
    slots = (cell_index * direction_bins)[None] + grab(direction) + (np.arange(count) * bin_count)[:, None, None]
    hist = np.bincount(slots.ravel(), weights=grab(magnitude).ravel(), minlength=count * bin_count)
    hist = hist.reshape(count, bin_count)
    hist = hist / (np.linalg.norm(hist, axis=1, keepdims=True) + 1e-3)

    # a small normalised copy of each patch keeps its overall shape
    small = patches.reshape(count, -1) @ shrink.T
    small = (small - small.mean(1, keepdims=True)) / (small.std(1, keepdims=True) + 1e-3)

    # size, shape and heat of each warm spot
    blob = np.zeros((count, 9))
    for i, (x, y, w, h, area) in enumerate(stats.tolist()):
        spot = contrast[y:y + h, x:x + w]
        blob[i] = [w, h, area, area / (w * h), w / h, spot.max(), spot.mean(), gray[y:y + h, x:x + w].max(), 0]
    # spread of brightness across each patch, worked out directly since numpy's std is slow here
    flat = patches.reshape(count, -1)
    average = flat.mean(1)
    blob[:, 8] = np.sqrt(np.maximum((flat * flat).mean(1) - average * average, 0))

    # overall brightness tells day from night
    frame_mean, frame_std = cv2.meanStdDev(gray)
    scene = np.tile([float(frame_mean[0, 0]), float(frame_std[0, 0])], (count, 1))

    rows = np.hstack([hist, small, blob, scene]).astype(np.float32)
    return rows, stats[:, :4].tolist()


def suppress_overlaps(detections, overlap):
    # keeps the best scoring box where boxes overlap by more than the limit
    if not detections:
        return []
    # stable sort so equal scores keep their order
    order = sorted(range(len(detections)), key=lambda i: -detections[i][1])
    boxes = np.array([detections[i][0] for i in order], np.float64)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = x0 + boxes[:, 2], y0 + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

    alive = np.ones(len(order), bool)
    kept = []
    for i in range(len(order)):
        if not alive[i]:
            continue
        kept.append(detections[order[i]])
        # overlap of this box with every box after it, the same sum as metrics.iou
        inter = (np.clip(np.minimum(x1[i], x1) - np.maximum(x0[i], x0), 0, None)
                 * np.clip(np.minimum(y1[i], y1) - np.maximum(y0[i], y0), 0, None))
        union = areas[i] + areas - inter
        ratio = np.where(union > 0, inter / np.where(union > 0, union, 1), 0)
        alive &= ~((ratio >= overlap) & (np.arange(len(order)) > i))
    return kept


def raw_scores(features, mean, std, w1, b1, w2, b2):
    # the network's output for each candidate before it is turned into a chance
    z = (features - mean) / std
    hidden = np.maximum(z @ w1 + b1, 0)
    return (hidden @ w2 + b2).ravel()


def to_chance(raw, scale, shift):
    # turns the network's output into a chance from 0 to 1
    # scale and shift are fitted on the validation images so the chance matches how often such hits are people
    return 1 / (1 + np.exp(-np.clip(scale * raw.astype(np.float64) + shift, -30, 30)))


def fitted_boxes(features, boxes, mean, std, box_weights):
    # moves and resizes each spot box to the person around it
    z = (features - mean) / std
    shift = np.hstack([z, np.ones((len(z), 1), np.float32)]) @ box_weights
    fitted = []
    for (x, y, w, h), (dx, dy, log_w, log_h) in zip(boxes, shift):
        new_w, new_h = float(np.exp(log_w)), float(np.exp(log_h))
        cx, cy = x + w / 2 + dx, y + h / 2 + dy
        fitted.append([cx - new_w / 2, cy - new_h / 2, new_w, new_h])
    return fitted


# loads the trained weights and finds people in 8 bit thermal frames
class person_check:
    def __init__(self, path, limit=None):
        weights = np.load(Path(path))
        self.mean, self.std = weights["mean"], weights["std"]
        self.w1, self.b1 = weights["w1"], weights["b1"]
        self.w2, self.b2 = weights["w2"], weights["b2"]
        self.box_weights = weights["box_weights"]
        self.threshold = float(weights["threshold"])
        self.overlap = float(weights["overlap"])
        # turns the network's raw output into a chance that matches how often such hits are right
        # older weight files have none, so their scores are used as they are
        self.scale = float(weights["scale"]) if "scale" in weights else 1.0
        self.shift = float(weights["shift"]) if "shift" in weights else 0.0
        # most candidates scored per frame, None scores them all
        self.limit = limit

        # weights made for a different set of features would give nonsense, so refuse them
        if self.mean.shape != (feature_count,) or self.w1.shape[0] != feature_count:
            raise ValueError(f"{path} does not match this version of the person check, "
                             "run scripts/train_person_check.py to make it again")

    def scores(self, features):
        # chance each candidate is a person, 0 to 1
        # calibrated on the validation images, so 0.8 means about 8 in 10 such hits were people there
        raw = raw_scores(features, self.mean, self.std, self.w1, self.b1, self.w2, self.b2)
        return to_chance(raw, self.scale, self.shift)

    def fit_boxes(self, features, boxes):
        # moves and resizes each spot box to the person around it
        return fitted_boxes(features, boxes, self.mean, self.std, self.box_weights)

    def detect(self, gray, threshold=None):
        # returns (box, score) pairs for one frame
        # threshold None uses the one tuned on the validation images, 0 keeps every candidate for map
        features, boxes = frame_features(gray, self.limit)
        if not boxes:
            return []
        limit = self.threshold if threshold is None else threshold
        scores = self.scores(features)
        fitted = self.fit_boxes(features, boxes)
        found = [(box, float(s)) for box, s in zip(fitted, scores) if s >= limit]
        return suppress_overlaps(found, self.overlap)
