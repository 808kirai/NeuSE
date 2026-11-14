"""
runs a mindyolo person detector on the unit with mindspore lite
the model is trained on modelarts and converted to a .ms file, see training/README.md

the input is one square rgb image scaled 0 to 1, made by letterboxing the frame
its size, 640 or 960 for example, is read from the model, so it always matches what the model was trained at
the output, read from mindyolo's yolov8 head, is one row per candidate box
    x centre, y centre, width, height in input pixels, then a score from 0 to 1 for each class
class 0 is person, a model trained with bicycles as class 1 still reports people only

mindspore lite is only needed on the unit that runs the model
"""
from pathlib import Path

import cv2
import numpy as np

from src.person_check import suppress_overlaps

# grey used to pad the frame to a square, the same as mindyolo's letterbox
pad_value = 114


def letterbox(gray, size):
    # fits the frame inside a size by size square without stretching it
    # returns the network input and how to map boxes back to the frame
    height, width = gray.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if scale != 1 else gray
    left, top = (size - new_w) // 2, (size - new_h) // 2
    square = np.full((size, size), pad_value, np.uint8)
    square[top:top + new_h, left:left + new_w] = resized
    # the same grey in all three colour channels, scaled 0 to 1, channels first
    batch = np.repeat(square[None, None].astype(np.float32) / 255.0, 3, axis=1)
    return batch, scale, left, top


def lite_runner(path, threads=4, device="cpu"):
    # loads a .ms or .mindir model with mindspore lite and returns a function that runs one batch
    # device ascend runs it on an ascend npu, such as the orange pi aipro's, cpu works anywhere
    import mindspore_lite as mslite

    context = mslite.Context()
    if device == "ascend":
        context.target = ["ascend"]
        context.ascend.device_id = 0
    else:
        context.target = ["cpu"]
        context.cpu.thread_num = threads
    model = mslite.Model()
    # converted .ms files are mindir lite, older mindspore lite versions name them differently
    kind = getattr(mslite.ModelType, "MINDIR_LITE", mslite.ModelType.MINDIR)
    if Path(path).suffix.lower() == ".mindir":
        kind = mslite.ModelType.MINDIR
    model.build_from_file(str(path), kind, context)

    def run(batch):
        inputs = model.get_inputs()
        inputs[0].set_data_from_numpy(np.ascontiguousarray(batch, dtype=np.float32))
        return model.predict(inputs)[0].get_data_to_numpy()

    # the square input size the model was exported at, from its input shape of 1, 3, size, size
    shape = list(model.get_inputs()[0].shape)
    run.size = shape[-1] if len(shape) == 4 and shape[-1] > 0 else None
    return run


class mindyolo_detector:
    # finds people in 8 bit thermal frames with a trained mindyolo model
    # runner takes the network input and returns its first output, tests pass their own
    def __init__(self, path=None, threshold=0.25, overlap=0.6, size=None, runner=None, most=1000, device="cpu"):
        self.runner = runner or lite_runner(path, device=device)
        # the score a box needs to be reported, set from the validation images once the model is trained
        self.threshold = threshold
        # boxes overlapping more than this are the same person
        self.overlap = overlap
        # the model's own input size when it says, otherwise 640
        self.size = size or getattr(self.runner, "size", None) or 640
        # most boxes kept before overlaps are removed, the highest scoring ones
        self.most = most

    def detect(self, gray, threshold=None):
        # returns (box, score) pairs for one frame, boxes as x, y, width, height in frame pixels
        # threshold None uses the model's own, 0 keeps every box for scoring with map
        batch, scale, left, top = letterbox(gray, self.size)
        output = np.asarray(self.runner(batch), dtype=np.float32)

        # some yolov8 exports put the box numbers first, as (1, 4 + classes, boxes), so turn those round
        # a 640 input gives thousands of boxes and only a handful of numbers per box, so the longer side is boxes
        if output.ndim == 3 and output.shape[0] == 1 and 5 <= output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)
        # the output must be one image of boxes with four box numbers and at least one class score
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] < 5:
            raise ValueError(f"expected a mindyolo output shaped (1, boxes, 4 + classes), got {output.shape}, "
                             "check the export in training/README.md")
        rows = output[0]
        scores = rows[:, 4]
        limit = self.threshold if threshold is None else threshold
        keep = np.flatnonzero(scores >= limit)
        if len(keep) > self.most:
            keep = keep[np.argsort(-scores[keep])[:self.most]]

        found = []
        height, width = gray.shape[:2]
        for i in keep:
            cx, cy, w, h = rows[i, :4]
            # undo the letterbox to get back to the frame's own pixels
            x0 = (cx - w / 2 - left) / scale
            y0 = (cy - h / 2 - top) / scale
            x1 = (cx + w / 2 - left) / scale
            y1 = (cy + h / 2 - top) / scale
            x0, y0 = max(0.0, float(x0)), max(0.0, float(y0))
            x1, y1 = min(float(width), float(x1)), min(float(height), float(y1))
            if x1 > x0 and y1 > y0:
                found.append(([x0, y0, x1 - x0, y1 - y0], float(scores[i])))
        return suppress_overlaps(found, self.overlap)
