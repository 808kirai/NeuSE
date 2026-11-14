"""
trains the person cnn with mindspore on an ordinary cpu
learns from the hit-uav training images, then uses the validation images to calibrate the scores and pick the
score a hit needs to be reported, the test images are never touched here

it needs mindspore, which is kept out of requirements.txt because the unit does not need it
    python -m venv C:/Users/you/venvs/neuse-mindspore
    C:/Users/you/venvs/neuse-mindspore/Scripts/python -m pip install mindspore==2.11.0 opencv-python==5.0.0.93
    C:/Users/you/venvs/neuse-mindspore/Scripts/python scripts/train_person_cnn.py --dataset C:/Users/you/datasets/hit-uav

it writes models/person_cnn.npz for the unit and models/person_cnn.mindir for mindspore lite
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# put the project root and this folder on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.hit_uav import load_split
from src.metrics import center_inside
from src.person_cnn import forward, inputs, layout_version, scalar_count
from train_person_check import by_image, calibrate, pick_threshold

models = Path(__file__).resolve().parent.parent / "models"
# boxes overlapping more than this are the same person
overlap = 0.5
# scalar columns that swap or invert when a patch is turned on its side, width and height, then aspect
width_column, height_column, aspect_column = 0, 1, 4


def collect(records, keep_negatives=1.0, seed=0):
    # network inputs for the candidates in every image, whether each sits on a person, and its box target
    # keep_negatives below 1 keeps only that share of spots that are not people, to save memory when training
    rng = np.random.default_rng(seed)
    patches, scalars, labels, targets, owners, boxes = [], [], [], [], [], []
    for index, record in enumerate(records):
        gray = cv2.imread(str(record["path"]), cv2.IMREAD_GRAYSCALE)
        p, s, spots = inputs(gray)
        for i, box in enumerate(spots):
            person = next((q for q in record["people"] if center_inside(box, q)), None)
            if person is None and keep_negatives < 1 and rng.random() > keep_negatives:
                continue
            x, y, w, h = box
            patches.append(p[i].astype(np.float16))
            scalars.append(s[i])
            labels.append(1.0 if person else 0.0)
            # where the person's centre is from the spot's centre, and the person's size as logs
            targets.append([person[0] + person[2] / 2 - (x + w / 2), person[1] + person[3] / 2 - (y + h / 2),
                            np.log(person[2]), np.log(person[3])] if person else [0, 0, 0, 0])
            owners.append(index)
            boxes.append(box)
    return (np.array(patches, np.float16).reshape(-1, 2, 32, 32), np.array(scalars, np.float32).reshape(-1, scalar_count),
            np.array(labels, np.float32), np.array(targets, np.float32).reshape(-1, 4), owners, boxes)


def turn(patches, scalars, targets, flip_x, flip_y, swap):
    # one of the eight ways to flip or turn a patch, a person seen from above can face any way
    # the box target and the width and height numbers move with the patch
    p, s, t = patches.copy(), scalars.copy(), targets.copy()
    if flip_x:
        p, t[:, 0] = p[..., ::-1], -t[:, 0]
    if flip_y:
        p, t[:, 1] = p[..., ::-1, :], -t[:, 1]
    if swap:
        p = p.swapaxes(-1, -2)
        t[:, [0, 1, 2, 3]] = t[:, [1, 0, 3, 2]]
        s[:, [width_column, height_column]] = s[:, [height_column, width_column]]
        s[:, aspect_column] = 1 / s[:, aspect_column]
    return np.ascontiguousarray(p), s, t


def build(scalar_mean, scalar_std):
    # the network, written so its layers match src/person_cnn.py exactly
    import mindspore as ms
    from mindspore import nn, ops

    class network(nn.Cell):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(2, 16, 3, stride=2, pad_mode="pad", padding=1, has_bias=True)
            self.conv2 = nn.Conv2d(16, 32, 3, pad_mode="pad", padding=1, has_bias=True)
            self.conv3 = nn.Conv2d(32, 64, 3, pad_mode="pad", padding=1, has_bias=True)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.dense1 = nn.Dense(64 * 4 * 4 + scalar_count, 64)
            self.dense2 = nn.Dense(64, 5)
            self.scalar_mean = ms.Tensor(scalar_mean, ms.float32)
            self.scalar_std = ms.Tensor(scalar_std, ms.float32)

        def construct(self, patches, scalars):
            x = self.relu(self.conv1(patches))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = ops.concat((self.flatten(x), (scalars - self.scalar_mean) / self.scalar_std), 1)
            return self.dense2(self.relu(self.dense1(x)))

    return network()


def weights_of(net, scalar_mean, scalar_std):
    # the trained numbers in the layout src/person_cnn.py reads
    out = {p.name: p.asnumpy().astype(np.float32) for p in net.get_parameters()}
    out = {name: value for name, value in out.items() if name.split(".")[0] in ("conv1", "conv2", "conv3", "dense1", "dense2")}
    out["scalar_mean"], out["scalar_std"] = scalar_mean, scalar_std
    return out


def train(dataset, out=None, epochs=12, negatives_per_person=4, batch=256, seed=0, export=True):
    import mindspore as ms
    from mindspore import nn, ops

    ms.set_seed(seed)
    rng = np.random.default_rng(seed)
    start = time.time()
    out = Path(out) if out else models / "person_cnn.npz"

    patches, scalars, labels, targets, _, _ = collect(load_split(dataset, "train"), keep_negatives=0.4, seed=seed)
    val_records = load_split(dataset, "val")
    val_patches, val_scalars, _, _, val_owners, val_boxes = collect(val_records)
    print(f"{len(labels)} training spots kept, {int(labels.sum())} on people, {len(val_boxes)} validation spots, "
          f"in {time.time() - start:.0f} s")

    scalar_mean = scalars.mean(0).astype(np.float32)
    scalar_std = (scalars.std(0) + 1e-6).astype(np.float32)
    people = labels == 1
    box_mean = targets[people].mean(0).astype(np.float32)
    box_std = (targets[people].std(0) + 1e-6).astype(np.float32)

    net = build(scalar_mean, scalar_std)
    steps_per_epoch = int(np.ceil(people.sum() * (1 + negatives_per_person) / batch))
    rates = nn.cosine_decay_lr(1e-5, 2e-3, steps_per_epoch * epochs, steps_per_epoch, epochs)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=rates, weight_decay=1e-4)
    score_loss = nn.BCEWithLogitsLoss()

    def forward_fn(p, s, y, t):
        raw = net(p, s)
        # the box only counts for spots on people
        on_person = y.reshape(-1, 1)
        box_error = ops.smooth_l1_loss(raw[:, 1:5], t, beta=1.0, reduction="none") * on_person
        box_loss = box_error.sum() / (on_person.sum() * 4 + 1)
        return score_loss(raw[:, 0], y) + box_loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    @ms.jit
    def step(p, s, y, t):
        loss, grads = grad_fn(p, s, y, t)
        optimizer(grads)
        return loss

    positives = np.flatnonzero(people)
    negatives = np.flatnonzero(~people)
    scaled_targets = (targets - box_mean) / box_std
    for epoch in range(epochs):
        # every person each round, with a fresh draw of spots that are not people
        chosen = np.concatenate([positives, rng.choice(negatives, len(positives) * negatives_per_person, replace=False)])
        rng.shuffle(chosen)
        total = 0.0
        for begin in range(0, len(chosen), batch):
            part = chosen[begin:begin + batch]
            p, s, t = turn(patches[part].astype(np.float32), scalars[part], scaled_targets[part],
                           rng.random() < 0.5, rng.random() < 0.5, rng.random() < 0.5)
            loss = step(ms.Tensor(p), ms.Tensor(s), ms.Tensor(labels[part]), ms.Tensor(t))
            total += float(loss.asnumpy()) * len(part)
        print(f"round {epoch + 1}/{epochs}, loss {total / len(chosen):.4f}, {time.time() - start:.0f} s")

    net.set_train(False)
    weights = weights_of(net, scalar_mean, scalar_std)

    # the numpy version must give the same numbers as mindspore, or the unit would score differently
    sample = slice(0, min(512, len(val_boxes)))
    expected = net(ms.Tensor(val_patches[sample].astype(np.float32)), ms.Tensor(val_scalars[sample])).asnumpy()
    got = forward(weights, val_patches[sample].astype(np.float32), val_scalars[sample])
    worst = float(np.abs(expected - got).max())
    print(f"numpy and mindspore agree to {worst:.2e}")
    if worst > 1e-3:
        raise RuntimeError("the numpy network does not match mindspore, check src/person_cnn.py")

    # calibrate the scores and pick the threshold on the validation images, the same way as the person check
    raw = np.concatenate([forward(weights, val_patches[i:i + 1024].astype(np.float32), val_scalars[i:i + 1024])
                          for i in range(0, len(val_boxes), 1024)])
    shift = raw[:, 1:5] * box_std + box_mean
    fitted = []
    for (x, y, w, h), (dx, dy, log_w, log_h) in zip(val_boxes, shift):
        new_w, new_h = float(np.exp(log_w)), float(np.exp(log_h))
        fitted.append([x + w / 2 + dx - new_w / 2, y + h / 2 + dy - new_h / 2, new_w, new_h])
    scale, offset = calibrate(val_records, raw[:, 0], fitted, val_owners)
    chances = 1 / (1 + np.exp(-np.clip(scale * raw[:, 0].astype(np.float64) + offset, -30, 30)))
    f1, threshold = pick_threshold(val_records, chances, fitted, val_owners)
    print(f"validation f1 {f1:.3f} at threshold {threshold:.3f}, calibration scale {scale:.3f} shift {offset:.3f}")

    np.savez(out, layout_version=np.int64(layout_version), box_mean=box_mean, box_std=box_std,
             scale=np.float64(scale), shift=np.float64(offset), threshold=np.float64(threshold),
             overlap=np.float32(overlap), **weights)
    print(f"saved {out}")

    if export:
        # the same network for mindspore lite, one spot at a time, mindspore lite can resize the batch
        ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
        ms.export(net, ms.Tensor(np.zeros((1, 2, 32, 32), np.float32)), ms.Tensor(np.zeros((1, scalar_count), np.float32)),
                  file_name=str(out.with_suffix("")), file_format="MINDIR")
        print(f"exported {out.with_suffix('.mindir')} in {time.time() - start:.0f} s")
    return f1, threshold


def main():
    parser = argparse.ArgumentParser(description="train the person cnn with mindspore on hit-uav")
    parser.add_argument("--dataset", required=True, help="folder of the cloned hit-uav repository")
    parser.add_argument("--out", help="where to save the weights, defaults to models/person_cnn.npz")
    parser.add_argument("--epochs", type=int, default=12, help="rounds through every person in the training images")
    parser.add_argument("--no-export", action="store_true", help="skip the mindir export")
    args = parser.parse_args()
    train(args.dataset, args.out, args.epochs, export=not args.no_export)


if __name__ == "__main__":
    main()
