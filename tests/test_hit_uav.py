import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_hit_uav import evaluate
from prepare_hit_uav import prepare
from src.config import config
from src.hit_uav import load_split, parse_name, splits


def make_dataset(root):
    # builds a tiny fake copy of hit-uav with the same folder layout and json format
    # each split has one night image with a warm person and a car, and one empty day image
    for split in splits:
        folder = root / "normal_json" / split
        folder.mkdir(parents=True)
        night, empty = "1_60_90_0_00001.jpg", "0_130_30_0_00002.jpg"

        frame = np.full((512, 640), 40, dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (114, 120), 255, -1)
        cv2.imwrite(str(folder / night), frame)
        cv2.imwrite(str(folder / empty), np.full((512, 640), 40, dtype=np.uint8))

        data = {
            "images": [
                {"filename": night, "height": 512, "width": 640, "id": 0},
                {"filename": empty, "height": 512, "width": 640, "id": 1},
            ],
            "annotation": [
                {"image_id": 0, "bbox": [100, 100, 15, 21], "category_id": 0, "id": 0},
                {"image_id": 0, "bbox": [300, 300, 30, 30], "category_id": 1, "id": 1},
                {"image_id": 0, "bbox": [500, 400, 20, 20], "category_id": 4, "id": 2},
            ],
        }
        annotations = root / "normal_json" / "annotations"
        annotations.mkdir(exist_ok=True)
        (annotations / f"{split}.json").write_text(json.dumps(data))
    return root


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path / "hit-uav")


def test_parse_name():
    assert parse_name("1_130_90_0_04077.jpg") == {"time": "night", "altitude": 130, "angle": 90}
    assert parse_name("0_60_30_0_00001.jpg")["time"] == "day"


def test_load_split_sorts_labels_by_kind(dataset):
    records = load_split(dataset, "test")
    night = next(r for r in records if r["time"] == "night")
    assert night["people"] == [[100, 100, 15, 21]]
    assert night["others"] == [[300, 300, 30, 30]]
    assert night["ignore"] == [[500, 400, 20, 20]]


def test_prepare_keeps_only_people(dataset, tmp_path):
    out = tmp_path / "person"
    summary = prepare(dataset, out, data_root="/data/person")
    assert summary["train"] == {"images": 2, "with_people": 1, "people": 1}

    label = (out / "labels" / "train" / "1_60_90_0_00001.txt").read_text().split()
    assert label[0] == "0"
    # centre of the person box scaled to the image
    # labels are saved to 6 decimal places
    assert float(label[1]) == pytest.approx(107.5 / 640, abs=1e-6)
    assert float(label[2]) == pytest.approx(110.5 / 512, abs=1e-6)

    # an image with no people still gets an empty label file
    assert (out / "labels" / "train" / "0_130_30_0_00002.txt").read_text() == ""
    assert (out / "images" / "val" / "0_130_30_0_00002.jpg").exists()
    assert "./images/test/1_60_90_0_00001.jpg" in (out / "test.txt").read_text()

    yaml = (out / "hit_uav_person.yaml").read_text()
    assert "train_set: /data/person/train.txt" in yaml
    assert "nc: 1" in yaml


def test_evaluate_finds_the_fake_person(dataset):
    result = evaluate(load_split(dataset, "test"), min_blob_area=60)
    center = result["scores"]["center"]["all"]
    assert center["tp"] == 1 and center["fn"] == 0
    assert result["scores"]["center"]["night"]["recall"] == 1.0


def test_evaluate_puts_settings_back(dataset):
    before = config.thermal_threshold, config.min_blob_area, config.clahe_clip_limit
    evaluate(load_split(dataset, "test"), thermal_threshold=123, min_blob_area=7, clahe_clip_limit=4.0)
    assert (config.thermal_threshold, config.min_blob_area, config.clahe_clip_limit) == before


def test_evaluate_rejects_unknown_settings(dataset):
    with pytest.raises(ValueError):
        evaluate(load_split(dataset, "test"), thermal_treshold=123)


def test_evaluate_reports_map(dataset):
    result = evaluate(load_split(dataset, "test"), min_blob_area=60)
    assert set(result["map50"]) == {"all", "day", "night", "60-80 m", "120-130 m"}
    assert 0.0 <= result["map50"]["all"] <= 1.0


def test_save_vis_writes_worst_frames(dataset, tmp_path):
    vis = tmp_path / "vis"
    evaluate(load_split(dataset, "test"), vis_dir=vis, vis_count=1, min_blob_area=60)
    saved = list(vis.glob("*.jpg"))
    assert len(saved) == 1
    # raw and marked frames side by side
    assert cv2.imread(str(saved[0])).shape == (512, 1280, 3)


def test_evaluate_person_check_uses_its_threshold(dataset):
    result = evaluate(load_split(dataset, "test"), detector="person-check")
    assert result["detector"] == "person-check"
    assert 0 < result["cutoff"] < 1
    assert 0.0 <= result["map50"]["all"] <= 1.0


def test_bicycles_can_be_kept_as_their_own_class(dataset, tmp_path):
    # the dataset labels a bike and its rider as one bicycle box, with no person box for the rider
    path = dataset / "normal_json" / "annotations" / "train.json"
    data = json.loads(path.read_text())
    data["annotation"].append({"image_id": 0, "bbox": [400, 100, 20, 30], "category_id": 2, "id": 9})
    path.write_text(json.dumps(data))

    night = next(r for r in load_split(dataset, "train") if r["time"] == "night")
    assert night["bicycles"] == [[400, 100, 20, 30]] and [400, 100, 20, 30] in night["others"]

    plain = tmp_path / "plain"
    prepare(dataset, plain)
    assert len((plain / "labels" / "train" / "1_60_90_0_00001.txt").read_text().splitlines()) == 1
    assert "nc: 1" in (plain / "hit_uav_person.yaml").read_text()

    both = tmp_path / "both"
    prepare(dataset, both, with_bicycles=True)
    lines = (both / "labels" / "train" / "1_60_90_0_00001.txt").read_text().splitlines()
    assert [line.split()[0] for line in lines] == ["0", "1"]
    assert "nc: 2" in (both / "hit_uav_person.yaml").read_text()
