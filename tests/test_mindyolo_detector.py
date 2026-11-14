"""
checks the mindyolo slot with a stand in for the model, since mindspore lite is only on units that run it
the stand in returns the output shape read from mindyolo's yolov8 head, one row per box
    x centre, y centre, width, height in input pixels, then a score per class
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import src.pipeline as pipeline_module
from src.config import config
from src.mindyolo_detector import letterbox, mindyolo_detector
from src.pipeline import thermal_pipeline


def fake_model(rows, size=640):
    # a runner that checks the input it gets and answers with fixed rows
    def run(batch):
        assert batch.shape == (1, 3, size, size) and batch.dtype == np.float32
        assert 0.0 <= batch.min() and batch.max() <= 1.0
        return np.array([rows], np.float32)
    return run


# a 640 by 512 frame sits in the 640 square with 64 grey rows above and below it
# so a person at frame x 100 to 114 and y 100 to 115 has its centre at 107, 171.5 in the square
person = [107.0, 171.5, 14.0, 15.0, 0.9]


def test_letterbox_keeps_the_frame_whole():
    frame = np.full((512, 640), 200, np.uint8)
    batch, scale, left, top = letterbox(frame, 640)
    assert (scale, left, top) == (1.0, 0, 64)
    assert batch[0, 0, 64, 0] == pytest.approx(200 / 255) and batch[0, 0, 0, 0] == pytest.approx(114 / 255)
    # all three colour channels are the same grey
    assert (batch[0, 0] == batch[0, 2]).all()

    big = np.zeros((1024, 1280), np.uint8)
    _, scale, left, top = letterbox(big, 640)
    assert (scale, left, top) == (0.5, 0, 64)


def test_boxes_come_back_in_the_frame_s_own_pixels():
    detector = mindyolo_detector(runner=fake_model([person]))
    found = detector.detect(np.zeros((512, 640), np.uint8))
    assert len(found) == 1
    box, score = found[0]
    assert box == pytest.approx([100, 100, 14, 15]) and score == pytest.approx(0.9)


def test_a_bigger_frame_is_scaled_back_up():
    detector = mindyolo_detector(runner=fake_model([person]))
    box, _ = detector.detect(np.zeros((1024, 1280), np.uint8))[0]
    # half scale, with 64 grey rows in the square standing for 128 frame rows
    assert box == pytest.approx([200, 200, 28, 30])


def test_low_scores_and_overlapping_boxes_are_dropped():
    rows = [person, [108.0, 172.0, 14.0, 15.0, 0.8], [300.0, 300.0, 10.0, 10.0, 0.1]]
    detector = mindyolo_detector(runner=fake_model(rows), threshold=0.25)
    found = detector.detect(np.zeros((512, 640), np.uint8))
    assert [score for _, score in found] == pytest.approx([0.9])
    # threshold 0 keeps every box that does not overlap a better one, for scoring with map
    assert len(detector.detect(np.zeros((512, 640), np.uint8), threshold=0.0)) == 2


def test_only_the_person_class_is_reported():
    # a model trained with bicycles as class 1 has a second score column
    rows = [[107.0, 171.5, 14.0, 15.0, 0.05, 0.95], [207.0, 171.5, 14.0, 15.0, 0.7, 0.1]]
    found = mindyolo_detector(runner=fake_model(rows)).detect(np.zeros((512, 640), np.uint8))
    assert len(found) == 1 and found[0][0][0] == pytest.approx(200)


def test_boxes_in_the_grey_border_are_trimmed_or_dropped():
    rows = [[107.0, 20.0, 14.0, 15.0, 0.9], [107.0, 62.0, 14.0, 10.0, 0.9]]
    found = mindyolo_detector(runner=fake_model(rows)).detect(np.zeros((512, 640), np.uint8))
    # the first lies wholly in the border above the frame, the second reaches 3 pixels into it
    assert len(found) == 1 and found[0][0] == pytest.approx([100, 0, 14, 3])


def test_boxes_given_numbers_first_are_turned_round():
    # some exports give (1, 4 + classes, boxes) instead of (1, boxes, 4 + classes)
    rows = np.zeros((8400, 5), np.float32)
    rows[0] = person
    detector = mindyolo_detector(runner=lambda batch: rows.T[None])
    box, score = detector.detect(np.zeros((512, 640), np.uint8))[0]
    assert box == pytest.approx([100, 100, 14, 15]) and score == pytest.approx(0.9)


@pytest.mark.parametrize("shape", [(8400, 5), (2, 8400, 5), (1, 8400, 4)])
def test_an_output_of_the_wrong_shape_is_named(shape):
    detector = mindyolo_detector(runner=lambda batch: np.zeros(shape, np.float32))
    with pytest.raises(ValueError, match="expected a mindyolo output"):
        detector.detect(np.zeros((512, 640), np.uint8))


def test_pipeline_runs_the_mindyolo_detector(isolated_db, monkeypatch, tmp_path):
    model = tmp_path / "person.ms"
    model.write_bytes(b"stand in")
    monkeypatch.setattr(config, "mindyolo_path", model)
    monkeypatch.setattr(pipeline_module, "mindyolo_detector",
                        lambda path, threshold, device: mindyolo_detector(runner=fake_model([person]),
                                                                          threshold=threshold))
    pipe = thermal_pipeline("mindyolo")
    frame = np.full((512, 640), 40, np.uint8)
    frame[100:115, 100:114] = 220
    hits = pipe.process_frames(thermal_frame=frame)
    assert pipe.detector == "mindyolo" and len(hits) == 1
    assert hits[0]["bbox"] == [100, 100, 14, 15] and hits[0]["detector"] == "mindyolo"
    assert hits[0]["confidence"] == pytest.approx(0.9) and hits[0]["thermal_peak"] == 220


def test_a_missing_model_falls_back_to_the_person_check(isolated_db, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "mindyolo_path", tmp_path / "missing.ms")
    assert thermal_pipeline("mindyolo").detector == "person-check"


def test_missing_mindspore_lite_falls_back_to_the_person_check(isolated_db, monkeypatch, tmp_path):
    model = tmp_path / "person.ms"
    model.write_bytes(b"stand in")
    monkeypatch.setattr(config, "mindyolo_path", model)
    # mindspore lite is not installed on this computer or in ci, so loading the real runner fails
    monkeypatch.setitem(sys.modules, "mindspore_lite", None)
    assert thermal_pipeline("mindyolo").detector == "person-check"


def test_the_scoring_script_takes_the_mindyolo_threshold(isolated_db, monkeypatch, tmp_path):
    from evaluate_hit_uav import make_detector

    model = tmp_path / "person.ms"
    model.write_bytes(b"stand in")
    monkeypatch.setattr(config, "mindyolo_path", model)
    monkeypatch.setattr(config, "mindyolo_threshold", 0.4)
    monkeypatch.setattr(pipeline_module, "mindyolo_detector",
                        lambda path, threshold, device: mindyolo_detector(runner=fake_model([person]),
                                                                          threshold=threshold))
    detect, cutoff, name = make_detector("mindyolo")
    assert name == "mindyolo" and cutoff == 0.4
    assert detect(np.zeros((512, 640), np.uint8))[0][1] == pytest.approx(0.9)


class fake_lite:
    # stands in for mindspore lite, recording where the model was asked to run
    class Context:
        def __init__(self):
            self.target = None
            self.cpu = type("cpu", (), {})()
            self.ascend = type("ascend", (), {})()

    class ModelType:
        MINDIR = "mindir"
        MINDIR_LITE = "mindir_lite"

    class Model:
        built = []

        def __init__(self):
            self.input = type("tensor", (), {"shape": [1, 3, 960, 960], "set_data_from_numpy": lambda s, data: None})()

        def build_from_file(self, path, kind, context):
            fake_lite.Model.built.append((kind, context))

        def get_inputs(self):
            return [self.input]

        def predict(self, inputs):
            rows = np.array([[[160.5, 257.25, 21.0, 22.5, 0.9]]], np.float32)
            return [type("out", (), {"get_data_to_numpy": lambda s: rows})()]


@pytest.mark.parametrize("device", ["cpu", "ascend"])
def test_the_model_runs_where_asked_at_its_own_input_size(monkeypatch, tmp_path, device):
    monkeypatch.setitem(sys.modules, "mindspore_lite", fake_lite)
    fake_lite.Model.built = []
    detector = mindyolo_detector(tmp_path / "person.ms", device=device)
    kind, context = fake_lite.Model.built[0]
    assert context.target == [device] and kind == "mindir_lite"
    if device == "ascend":
        assert context.ascend.device_id == 0
    # a model exported at 960 gets a 960 square, with the 640 by 512 frame scaled up 1.5 times
    assert detector.size == 960
    box, _ = detector.detect(np.zeros((512, 640), np.uint8))[0]
    assert box == pytest.approx([100, 100, 14, 15])
