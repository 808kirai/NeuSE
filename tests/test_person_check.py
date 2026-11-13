import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.config import config
from src.hit_uav import splits
from src.metrics import iou
from src.person_check import (feature_count, find_candidates, frame_features, person_check, shrink,
                              suppress_overlaps)
from src.pipeline import person_check_name, thermal_pipeline


def drone_frame(seed=0):
    # a 640 by 512 frame like hit-uav, mild noise with a few person sized warm spots and one tiny speck
    rng = np.random.default_rng(seed)
    frame = rng.integers(30, 50, (512, 640), dtype=np.uint8)
    for x, y in ((100, 100), (300, 250), (500, 400)):
        cv2.ellipse(frame, (x, y), (6, 8), 0, 0, 360, 200, -1)
    frame[50, 50] = 255
    return frame


def test_candidates_are_person_sized_warm_spots():
    candidates, _ = find_candidates(drone_frame())
    # the three spots are found and the single hot pixel is not
    assert len(candidates) == 3
    assert all(15 <= stats[4] <= 800 for stats, _ in candidates)


def test_flat_frame_has_no_candidates():
    features, boxes = frame_features(np.full((512, 640), 40, np.uint8))
    assert features.shape == (0, feature_count) and boxes == []


def test_features_have_one_row_per_candidate():
    features, boxes = frame_features(drone_frame())
    assert features.shape == (3, feature_count)
    assert np.isfinite(features).all()


def test_spot_on_the_frame_edge_gets_full_features():
    frame = np.full((512, 640), 40, np.uint8)
    cv2.ellipse(frame, (0, 0), (6, 8), 0, 0, 360, 200, -1)
    features, boxes = frame_features(frame)
    assert len(boxes) == 1 and np.isfinite(features).all()


def test_shrink_matches_opencv():
    patch = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    expected = cv2.resize(patch, (12, 12), interpolation=cv2.INTER_AREA).ravel()
    assert np.allclose(shrink @ patch.ravel(), expected, atol=1e-5)


def simple_suppress(detections, overlap):
    # plain loop version kept to check the fast one against
    kept = []
    for box, score in sorted(detections, key=lambda d: -d[1]):
        if all(iou(box, other) < overlap for other, _ in kept):
            kept.append((box, score))
    return kept


def test_suppress_overlaps_keeps_the_best_box():
    detections = [([0, 0, 10, 10], 0.6), ([1, 0, 10, 10], 0.9), ([50, 50, 10, 10], 0.5)]
    kept = suppress_overlaps(detections, 0.5)
    assert kept == [([1, 0, 10, 10], 0.9), ([50, 50, 10, 10], 0.5)]


def test_suppress_overlaps_matches_the_simple_loop():
    rng = np.random.default_rng(2)
    for _ in range(20):
        detections = [(rng.integers(0, 60, 2).tolist() + rng.integers(5, 20, 2).tolist(), float(rng.random()))
                      for _ in range(40)]
        assert suppress_overlaps(detections, 0.5) == simple_suppress(detections, 0.5)


def test_suppress_overlaps_with_nothing():
    assert suppress_overlaps([], 0.5) == []


def test_saved_weights_load_and_score():
    checker = person_check(config.person_check_path)
    features, _ = frame_features(drone_frame())
    scores = checker.scores(features)
    assert scores.shape == (3,)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert 0 < checker.threshold < 1


def test_pipeline_runs_the_person_check():
    pipe = thermal_pipeline("person-check")
    assert pipe.checker is not None
    # min_score 0 keeps every candidate so there is something to check
    hits = pipe._detect(drone_frame(), min_score=0.0)
    assert hits
    for hit in hits:
        assert hit["detector"] == person_check_name
        assert 0 <= hit["confidence"] <= 1
        x, y, w, h = hit["bbox"]
        assert x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= 640 and y + h <= 512


def test_person_check_gives_the_same_answer_twice():
    pipe = thermal_pipeline("person-check")
    assert pipe._detect(drone_frame(), 0.0) == pipe._detect(drone_frame(), 0.0)


def test_person_check_handles_bad_frames():
    pipe = thermal_pipeline("person-check")
    for bad in (None, np.array([]), np.zeros((10, 10, 4), np.uint8)):
        assert pipe.process_frames(thermal_frame=bad) == []
    assert pipe.process_frames(thermal_frame=np.zeros((512, 640), np.uint16)) == []


def test_missing_weights_fall_back_to_warm_spots(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "person_check_path", tmp_path / "missing.npz")
    assert thermal_pipeline("person-check").detector == "warm-spots"


def test_unknown_detector_is_refused():
    with pytest.raises(ValueError):
        thermal_pipeline("yolo")


def test_configured_detector_is_used(monkeypatch):
    monkeypatch.setattr(config, "detector", "person-check")
    assert thermal_pipeline().detector == "person-check"


def make_dataset(root):
    # tiny hit-uav copy with a warm person and a warm distractor in each night frame
    for split in splits:
        folder = root / "normal_json" / split
        folder.mkdir(parents=True)
        images, labels = [], []
        for i in range(3):
            name = f"1_60_90_0_{i:05d}.jpg"
            cv2.imwrite(str(folder / name), drone_frame(i))
            images.append({"filename": name, "height": 512, "width": 640, "id": i})
            # the first spot is labelled a person, the other two are not
            labels.append({"image_id": i, "bbox": [94, 92, 13, 17], "category_id": 0, "id": i})
        annotations = root / "normal_json" / "annotations"
        annotations.mkdir(exist_ok=True)
        (annotations / f"{split}.json").write_text(json.dumps({"images": images, "annotation": labels}))
    return root


def test_training_writes_usable_weights(tmp_path):
    from train_person_check import train

    out = tmp_path / "weights.npz"
    f1, threshold = train(make_dataset(tmp_path / "hit-uav"), out, epochs=2)
    assert 0 <= f1 <= 1
    checker = person_check(out)
    assert checker.threshold == pytest.approx(threshold)
    assert isinstance(checker.detect(drone_frame()), list)


def test_scores_are_calibrated_chances():
    checker = person_check(config.person_check_path)
    # the saved weights carry the calibration fitted on the validation images
    assert checker.scale > 0 and (checker.scale, checker.shift) != (1.0, 0.0)
    features, _ = frame_features(drone_frame())
    scores = checker.scores(features)
    assert ((scores >= 0) & (scores <= 1)).all()
    # calibration keeps the order of the scores, so the ranking and the map score do not change
    plain = checker.scores.__func__
    checker.scale, checker.shift = 1.0, 0.0
    assert (np.argsort(plain(checker, features)) == np.argsort(scores)).all()


def test_weights_for_other_features_are_refused(tmp_path):
    weights = dict(np.load(config.person_check_path))
    weights["mean"] = weights["mean"][:-1]
    np.savez(tmp_path / "old.npz", **weights)
    with pytest.raises(ValueError, match="does not match"):
        person_check(tmp_path / "old.npz")


def test_weights_without_calibration_still_load(tmp_path):
    weights = {k: v for k, v in np.load(config.person_check_path).items() if k not in ("scale", "shift")}
    np.savez(tmp_path / "plain.npz", **weights)
    checker = person_check(tmp_path / "plain.npz")
    assert (checker.scale, checker.shift) == (1.0, 0.0)


def test_a_noisy_frame_is_capped_to_the_warmest_candidates():
    noise = np.random.default_rng(3).integers(0, 255, (512, 640), dtype=np.uint8)
    everything, _ = frame_features(noise)
    capped, boxes = frame_features(noise, limit=50)
    assert len(everything) > 50 and capped.shape == (50, feature_count) and len(boxes) == 50
    # a frame under the cap is left alone
    assert len(frame_features(drone_frame(), limit=50)[1]) == 3


def test_frame_scale_brings_a_bigger_picture_down_to_the_trained_size(monkeypatch):
    small = drone_frame()
    # the same scene from a camera with twice the detail, where people look twice as big
    big = cv2.resize(small, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    expected = thermal_pipeline("person-check")._detect(small, min_score=0.0)

    monkeypatch.setattr(config, "frame_scale", 0.5)
    hits = thermal_pipeline("person-check")._detect(big, min_score=0.0)
    assert len(hits) == len(expected) > 0
    for hit, same in zip(hits, expected):
        # boxes come back in the big frame's own pixels
        assert hit["bbox"] == pytest.approx([2 * v for v in same["bbox"]], abs=2)
        assert hit["confidence"] == pytest.approx(same["confidence"], abs=1e-6)


def test_person_check_warns_once_about_an_unexpected_frame_size(caplog):
    pipe = thermal_pipeline("person-check")
    with caplog.at_level("WARNING"):
        pipe._detect(np.zeros((480, 640), np.uint8))
        pipe._detect(np.zeros((480, 640), np.uint8))
    assert sum("trained on 640 by 512" in r.message for r in caplog.records) == 1


def test_confidence_is_not_rounded_away():
    hits = thermal_pipeline("person-check")._detect(drone_frame(), min_score=0.0)
    assert all(isinstance(h["confidence"], float) for h in hits)
