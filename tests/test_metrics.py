import pytest

from src.metrics import add_counts, average_precision, center_inside, iou, match_image, summarize


def test_iou_same_box_is_one_and_apart_is_zero():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 10, 10]) == 0.0


def test_iou_half_overlap():
    # 50 shared pixels over 150 covered
    assert iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(1 / 3)


def test_center_inside():
    assert center_inside([8, 8, 4, 4], [0, 0, 20, 20])
    assert not center_inside([30, 30, 4, 4], [0, 0, 20, 20])


def test_one_good_hit_one_miss_one_false():
    people = [[10, 10, 10, 10], [100, 100, 10, 10]]
    dets = [[11, 11, 10, 10], [300, 300, 10, 10]]
    counts = match_image(dets, people)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "fp_on_other": 0, "ignored": 0}


def test_two_hits_on_one_person_count_once():
    people = [[10, 10, 10, 10]]
    dets = [[10, 10, 10, 10], [11, 10, 10, 10]]
    counts = match_image(dets, people)
    assert counts["tp"] == 1 and counts["fp"] == 1


def test_matching_does_not_depend_on_detection_order():
    # the first detection overlaps both people, the second only fits the first person
    people = [[0, 0, 10, 10], [6, 0, 10, 10]]
    dets = [[3, 0, 10, 10], [0, 0, 10, 10]]
    forward = match_image(dets, people)
    backward = match_image(dets[::-1], people)
    assert forward == backward
    assert forward["tp"] == 2


def test_hits_in_dont_care_regions_are_ignored():
    counts = match_image([[52, 52, 6, 6]], [], ignore=[[50, 50, 20, 20]])
    assert counts["fp"] == 0 and counts["ignored"] == 1


def test_false_hits_on_vehicles_are_counted():
    counts = match_image([[52, 52, 6, 6]], [], others=[[50, 50, 20, 20]])
    assert counts["fp"] == 1 and counts["fp_on_other"] == 1


def test_center_rule_accepts_a_small_blob_inside_a_person():
    # a warm core much smaller than the labelled box
    people = [[10, 10, 20, 20]]
    dets = [[17, 17, 6, 6]]
    assert match_image(dets, people, rule="iou")["tp"] == 0
    assert match_image(dets, people, rule="center")["tp"] == 1


def test_summarize():
    total = add_counts({}, {"tp": 3, "fp": 1, "fn": 1})
    m = summarize(total)
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["f1"] == 0.75


def test_summarize_with_nothing_found_or_labelled():
    m = summarize({})
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_map_is_one_when_every_person_is_found_with_no_false_hits():
    images = [([([0, 0, 10, 10], 0.9)], [[0, 0, 10, 10]], []), ([([5, 5, 10, 10], 0.8)], [[5, 5, 10, 10]], [])]
    assert average_precision(images) == pytest.approx(1.0)


def test_map_is_zero_when_nothing_matches():
    images = [([([100, 100, 10, 10], 0.9)], [[0, 0, 10, 10]], [])]
    assert average_precision(images) == 0.0


def test_map_rewards_ranking_people_above_false_hits():
    people = [[0, 0, 10, 10]]
    good_first = [([([0, 0, 10, 10], 0.9), ([50, 50, 10, 10], 0.1)], people, [])]
    bad_first = [([([0, 0, 10, 10], 0.1), ([50, 50, 10, 10], 0.9)], people, [])]
    assert average_precision(good_first) == pytest.approx(1.0)
    assert average_precision(bad_first) == pytest.approx(0.5)


def test_map_ignores_hits_in_dont_care_regions():
    images = [([([0, 0, 10, 10], 0.5), ([52, 52, 6, 6], 0.9)], [[0, 0, 10, 10]], [[50, 50, 20, 20]])]
    assert average_precision(images) == pytest.approx(1.0)


def test_map_with_no_people_is_zero():
    assert average_precision([([([0, 0, 10, 10], 0.9)], [], [])]) == 0.0
