"""
person cnn, a small convolutional network trained with mindspore that decides which warm spots are people
step 1 finds warm spots by local contrast at two sizes, so very small people are not missed
step 2 cuts a 32 by 32 patch around each spot and the network scores it and fits a box to the person
step 3 turns the score into a calibrated chance and drops overlapping boxes

the network is trained by scripts/train_person_cnn.py with mindspore on a laptop cpu
it runs here in plain numpy, so the unit needs no ai framework
the same network is also exported to mindir, for mindspore lite on an ascend board such as the orange pi aipro
"""
from pathlib import Path

import cv2
import numpy as np

from src.person_check import suppress_overlaps

# warm spot finding, two window sizes, each with the contrast a pixel needs above its surroundings
# the small window finds people only a few pixels across, measured on the validation images
# with only the large window 93.4% of people get a spot, with both 96.7%
spot_scales = ((15, 25), (9, 15))
min_area = 15
max_area = 800
# spots whose centres are this close in pixels are the same spot found at both sizes
same_spot = 4

patch_size = 32
half = patch_size // 2
# the contrast channel is divided by this so it sits near the same range as the scene channel
contrast_scale = 32.0
# blob numbers given to the network next to the patch
scalar_count = 11
# version of the patch and scalar layout, weights made for another layout are refused
layout_version = 1


def find_spots(gray):
    # returns (x, y, w, h, area) and centre for every warm spot, plus the local contrast map of the first size
    image = gray.astype(np.float32)
    spots, maps = [], []
    for window, needed in spot_scales:
        contrast = image - cv2.blur(image, (window, window))
        maps.append(contrast)
        _, _, stats, centres = cv2.connectedComponentsWithStats((contrast > needed).astype(np.uint8), connectivity=8)
        for s, c in zip(stats[1:].tolist(), centres[1:].tolist()):
            if not min_area <= s[4] <= max_area:
                continue
            if all(abs(c[0] - k[1][0]) > same_spot or abs(c[1] - k[1][1]) > same_spot for k in spots):
                spots.append((s, c))
    return spots, maps[0]


def inputs(gray, limit=None):
    # the network's inputs for every spot in a frame
    # returns patches shaped (spots, 2, 32, 32), scalars shaped (spots, 11) and each spot's box
    # channel 0 is the frame scaled by its own brightness and spread, so day and night look alike
    # channel 1 is the local contrast, how much warmer each pixel is than its surroundings
    spots, contrast = find_spots(gray)
    if not spots:
        return np.zeros((0, 2, patch_size, patch_size), np.float32), np.zeros((0, scalar_count), np.float32), []
    if limit and len(spots) > limit:
        # keep the warmest spots when a noisy frame has very many
        peaks = [float(contrast[y:y + h, x:x + w].max()) for (x, y, w, h, _), _ in spots]
        warmest = sorted(range(len(spots)), key=lambda i: -peaks[i])[:limit]
        spots = [spots[i] for i in sorted(warmest)]

    mean, spread = cv2.meanStdDev(gray)
    mean, spread = float(mean[0, 0]), float(spread[0, 0]) + 1e-3
    scene = (gray.astype(np.float32) - mean) / spread
    padded_scene = cv2.copyMakeBorder(scene, half, half, half, half, cv2.BORDER_REFLECT)
    padded_contrast = cv2.copyMakeBorder(contrast / contrast_scale, half, half, half, half, cv2.BORDER_REFLECT)

    count = len(spots)
    patches = np.zeros((count, 2, patch_size, patch_size), np.float32)
    scalars = np.zeros((count, scalar_count), np.float32)
    boxes = []
    for i, ((x, y, w, h, area), (cx, cy)) in enumerate(spots):
        # the patch is centred on the spot, in padded coordinates the centre moves by half a patch
        top, left = int(round(cy)), int(round(cx))
        patches[i, 0] = padded_scene[top:top + patch_size, left:left + patch_size]
        patches[i, 1] = padded_contrast[top:top + patch_size, left:left + patch_size]
        spot = contrast[y:y + h, x:x + w]
        scalars[i] = [w, h, area, area / (w * h), w / h, spot.max(), spot.mean(),
                      gray[y:y + h, x:x + w].max(), patches[i, 0].std(), mean, spread]
        boxes.append([x, y, w, h])
    return patches, scalars, boxes


def conv(x, weight, bias, stride=1):
    # a 3 by 3 convolution with one pixel of zero padding, the same as mindspore's Conv2d with pad_mode pad
    # x is channels last, (batch, height, width, in), and weight is mindspore's (out, in, 3, 3)
    # the nine shifted copies of x are laid side by side so the whole layer is one matrix multiply
    batch, height, width, channels = x.shape
    height, width = height // stride, width // stride
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    columns = np.empty((batch, height, width, 9, channels), np.float32)
    for i in range(3):
        for j in range(3):
            columns[:, :, :, i * 3 + j] = padded[:, i:i + stride * height:stride, j:j + stride * width:stride]
    # weight laid out the same way, kernel position first then input channel
    kernel = weight.transpose(2, 3, 1, 0).reshape(9 * channels, -1)
    out = columns.reshape(-1, 9 * channels) @ kernel
    return out.reshape(batch, height, width, -1) + bias


def pool(x):
    # 2 by 2 max pooling with stride 2, channels last
    batch, height, width, channels = x.shape
    return x.reshape(batch, height // 2, 2, width // 2, 2, channels).max(axis=(2, 4))


def forward(weights, patches, scalars):
    # the network's raw outputs, column 0 is the person score before calibration
    # columns 1 to 4 are the box shift and log size, the same layout as the training script
    x = np.ascontiguousarray(patches.astype(np.float32).transpose(0, 2, 3, 1))
    # the first layer steps two pixels at a time, which halves the patch to 16 by 16 more cheaply than pooling
    x = np.maximum(conv(x, weights["conv1.weight"], weights["conv1.bias"], stride=2), 0)
    for name in ("conv2", "conv3"):
        x = pool(np.maximum(conv(x, weights[f"{name}.weight"], weights[f"{name}.bias"]), 0))
    scaled = (scalars - weights["scalar_mean"]) / weights["scalar_std"]
    # back to channels first before flattening, the order mindspore's Flatten uses
    x = np.concatenate([x.transpose(0, 3, 1, 2).reshape(len(x), -1), scaled], axis=1)
    x = np.maximum(x @ weights["dense1.weight"].T + weights["dense1.bias"], 0)
    return x @ weights["dense2.weight"].T + weights["dense2.bias"]


class person_cnn:
    # loads the trained weights and finds people in 8 bit thermal frames
    def __init__(self, path, limit=None, batch=256):
        loaded = np.load(Path(path))
        if int(loaded["layout_version"]) != layout_version:
            raise ValueError(f"{path} was made for a different version of the person cnn, "
                             "run scripts/train_person_cnn.py to make it again")
        self.weights = {key: loaded[key].astype(np.float32) for key in loaded.files
                        if key not in ("layout_version", "scale", "shift", "threshold", "overlap", "box_mean", "box_std")}
        self.box_mean, self.box_std = loaded["box_mean"], loaded["box_std"]
        self.scale, self.shift = float(loaded["scale"]), float(loaded["shift"])
        self.threshold = float(loaded["threshold"])
        self.overlap = float(loaded["overlap"])
        self.limit = limit
        self.batch = batch

    def raw(self, patches, scalars):
        # runs the network in batches so a busy frame does not need much memory
        parts = [forward(self.weights, patches[i:i + self.batch], scalars[i:i + self.batch])
                 for i in range(0, len(patches), self.batch)]
        return np.concatenate(parts) if parts else np.zeros((0, 5), np.float32)

    def scores(self, raw):
        # calibrated on the validation images, so 0.8 means about 8 in 10 such hits were people there
        logit = np.clip(self.scale * raw[:, 0].astype(np.float64) + self.shift, -30, 30)
        return 1 / (1 + np.exp(-logit))

    def fit_boxes(self, raw, boxes):
        # moves and resizes each spot box to the person around it
        shift = raw[:, 1:5] * self.box_std + self.box_mean
        fitted = []
        for (x, y, w, h), (dx, dy, log_w, log_h) in zip(boxes, shift):
            new_w, new_h = float(np.exp(log_w)), float(np.exp(log_h))
            cx, cy = x + w / 2 + dx, y + h / 2 + dy
            fitted.append([cx - new_w / 2, cy - new_h / 2, new_w, new_h])
        return fitted

    def detect(self, gray, threshold=None):
        # returns (box, score) pairs for one frame
        # threshold None uses the one tuned on the validation images, 0 keeps every spot for map
        patches, scalars, boxes = inputs(gray, self.limit)
        if not boxes:
            return []
        raw = self.raw(patches, scalars)
        scores = self.scores(raw)
        fitted = self.fit_boxes(raw, boxes)
        limit = self.threshold if threshold is None else threshold
        found = [(box, float(s)) for box, s in zip(fitted, scores) if s >= limit]
        return suppress_overlaps(found, self.overlap)
