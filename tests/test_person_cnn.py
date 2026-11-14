"""
checks the person cnn's numpy network and its warm spot finding
the network is checked against a slow, plain version of the same maths, so it is right without mindspore
the training script also checks it against mindspore itself every time the model is trained
"""
import numpy as np
import pytest

import src.person_cnn as pc
from src.config import config
from src.pipeline import thermal_pipeline


def drone_frame(seed=0, spots=((100, 100), (300, 250), (500, 400))):
    rng = np.random.default_rng(seed)
    frame = rng.integers(30, 50, (512, 640)).astype(np.uint8)
    for x, y in spots:
        frame[y - 7:y + 8, x - 6:x + 7] = 200
    return frame


def random_weights(seed=0):
    rng = np.random.default_rng(seed)
    shapes = {"conv1.weight": (16, 2, 3, 3), "conv1.bias": (16,), "conv2.weight": (32, 16, 3, 3), "conv2.bias": (32,),
              "conv3.weight": (64, 32, 3, 3), "conv3.bias": (64,), "dense1.weight": (64, 64 * 4 * 4 + pc.scalar_count),
              "dense1.bias": (64,), "dense2.weight": (5, 64), "dense2.bias": (5,)}
    weights = {k: (rng.standard_normal(v) * 0.1).astype(np.float32) for k, v in shapes.items()}
    weights["scalar_mean"] = np.zeros(pc.scalar_count, np.float32)
    weights["scalar_std"] = np.ones(pc.scalar_count, np.float32)
    return weights


def slow_conv(x, weight, bias, stride=1):
    # the textbook convolution, channels last, one output pixel at a time
    batch, height, width, _ = x.shape
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    out = np.zeros((batch, height // stride, width // stride, weight.shape[0]), np.float32)
    for r in range(height // stride):
        for c in range(width // stride):
            window = padded[:, r * stride:r * stride + 3, c * stride:c * stride + 3, :]
            out[:, r, c] = np.einsum("bijc,ocij->bo", window, weight) + bias
    return out


@pytest.mark.parametrize("stride", [1, 2])
def test_fast_convolution_matches_the_textbook_one(stride):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 8, 8, 4)).astype(np.float32)
    weight = rng.standard_normal((5, 4, 3, 3)).astype(np.float32)
    bias = rng.standard_normal(5).astype(np.float32)
    assert np.allclose(pc.conv(x, weight, bias, stride), slow_conv(x, weight, bias, stride), atol=1e-4)


def test_pooling_keeps_the_largest_of_each_square():
    x = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    assert pc.pool(x)[0, :, :, 0].tolist() == [[5, 7], [13, 15]]


def test_network_gives_one_row_of_five_numbers_per_spot():
    patches, scalars, boxes = pc.inputs(drone_frame())
    raw = pc.forward(random_weights(), patches, scalars)
    assert raw.shape == (len(boxes), 5) and np.isfinite(raw).all()


def test_spots_are_found_at_two_sizes_and_not_doubled():
    patches, scalars, boxes = pc.inputs(drone_frame())
    # each warm spot once, though both window sizes find it
    assert len(boxes) == 3
    assert patches.shape == (3, 2, 32, 32) and scalars.shape == (3, pc.scalar_count)


def test_a_flat_frame_has_no_spots():
    patches, scalars, boxes = pc.inputs(np.full((512, 640), 40, np.uint8))
    assert boxes == [] and patches.shape == (0, 2, 32, 32)


def test_a_noisy_frame_is_capped_to_the_warmest_spots():
    noise = np.random.default_rng(3).integers(0, 255, (512, 640)).astype(np.uint8)
    _, _, every = pc.inputs(noise)
    _, _, capped = pc.inputs(noise, limit=40)
    assert len(every) > 40 and len(capped) == 40


def test_spots_at_the_frame_edge_get_full_patches():
    frame = np.full((512, 640), 40, np.uint8)
    frame[0:15, 0:13] = 200
    frame[497:512, 627:640] = 200
    patches, _, boxes = pc.inputs(frame)
    assert len(boxes) == 2 and np.isfinite(patches).all()


def save_weights(path, threshold=0.5):
    np.savez(path, layout_version=np.int64(pc.layout_version), box_mean=np.zeros(4, np.float32),
             box_std=np.ones(4, np.float32), scale=np.float64(1.0), shift=np.float64(0.0),
             threshold=np.float64(threshold), overlap=np.float32(0.5), **random_weights())


def test_detect_gives_boxes_and_chances(tmp_path):
    save_weights(tmp_path / "w.npz")
    model = pc.person_cnn(tmp_path / "w.npz")
    found = model.detect(drone_frame(), threshold=0.0)
    assert found and all(0 <= score <= 1 for _, score in found)
    assert all(len(box) == 4 for box, _ in found)


def test_weights_for_another_layout_are_refused(tmp_path):
    save_weights(tmp_path / "w.npz")
    weights = dict(np.load(tmp_path / "w.npz"))
    weights["layout_version"] = np.int64(pc.layout_version + 1)
    np.savez(tmp_path / "old.npz", **weights)
    with pytest.raises(ValueError, match="different version"):
        pc.person_cnn(tmp_path / "old.npz")


def test_pipeline_runs_the_person_cnn(isolated_db, monkeypatch, tmp_path):
    save_weights(tmp_path / "w.npz", threshold=0.0)
    monkeypatch.setattr(config, "person_cnn_path", tmp_path / "w.npz")
    pipe = thermal_pipeline("person-cnn")
    hits = pipe.process_frames(thermal_frame=drone_frame())
    assert pipe.detector == "person-cnn" and hits
    assert all(h["detector"] == "person-cnn" and 0 <= h["confidence"] <= 1 for h in hits)


def test_missing_cnn_weights_fall_back_to_the_person_check(isolated_db, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "person_cnn_path", tmp_path / "missing.npz")
    assert thermal_pipeline("person-cnn").detector == "person-check"
