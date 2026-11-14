"""
checks that model files are checked against their recorded sha-256 before the unit trusts them
a field unit can be lost or left unattended, and a swapped model could quietly stop finding people
"""
import shutil

import pytest

from src import integrity
from src.config import config
from src.pipeline import thermal_pipeline


def test_the_committed_weights_match_their_checksums():
    # catches weights retrained without their checksum being recorded
    folder = config.person_check_path.parent
    sums = integrity.listed(folder)
    for name in ("person_check.npz", "person_cnn.npz"):
        assert integrity.verify(folder / name) is True and sums[name] == integrity.sha256(folder / name)


@pytest.fixture
def copied(tmp_path):
    # copies of the real weights and their checksums, so they can be changed safely
    for name in ("person_check.npz", "person_cnn.npz", "checksums.txt"):
        shutil.copy(config.person_check_path.parent / name, tmp_path / name)
    return tmp_path


def damage(path):
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))


def test_a_changed_file_is_refused(copied):
    damage(copied / "person_check.npz")
    with pytest.raises(integrity.integrity_error, match="does not match"):
        integrity.verify(copied / "person_check.npz")


def test_a_file_with_no_checksum_loads_with_a_warning(copied, caplog):
    (copied / "checksums.txt").unlink()
    assert integrity.verify(copied / "person_check.npz") is False
    assert "no recorded checksum" in caplog.text


def test_a_checksum_given_directly_is_used(copied):
    path = copied / "person_cnn.npz"
    assert integrity.verify(path, integrity.sha256(path).upper()) is True
    with pytest.raises(integrity.integrity_error):
        integrity.verify(path, "0" * 64)


def test_recording_keeps_the_other_entries(copied):
    damage(copied / "person_cnn.npz")
    integrity.record(copied / "person_cnn.npz")
    sums = integrity.listed(copied)
    assert set(sums) == {"person_check.npz", "person_cnn.npz"}
    assert integrity.verify(copied / "person_cnn.npz") is True


def test_changed_person_check_weights_fall_back_to_warm_spots(isolated_db, monkeypatch, copied):
    damage(copied / "person_check.npz")
    monkeypatch.setattr(config, "person_check_path", copied / "person_check.npz")
    assert thermal_pipeline("person-check").detector == "warm-spots"


def test_damaged_unreadable_weights_also_fall_back(isolated_db, monkeypatch, tmp_path):
    (tmp_path / "person_check.npz").write_bytes(b"not a model")
    monkeypatch.setattr(config, "person_check_path", tmp_path / "person_check.npz")
    assert thermal_pipeline("person-check").detector == "warm-spots"


def test_changed_person_cnn_weights_fall_back_to_the_person_check(isolated_db, monkeypatch, copied):
    damage(copied / "person_cnn.npz")
    monkeypatch.setattr(config, "person_cnn_path", copied / "person_cnn.npz")
    assert thermal_pipeline("person-cnn").detector == "person-check"


def test_a_mindyolo_model_with_the_wrong_checksum_is_refused(isolated_db, monkeypatch, tmp_path):
    model = tmp_path / "person.ms"
    model.write_bytes(b"stand in")
    monkeypatch.setattr(config, "mindyolo_path", model)
    monkeypatch.setattr(config, "mindyolo_sha256", "0" * 64)
    assert thermal_pipeline("mindyolo").detector == "person-check"
