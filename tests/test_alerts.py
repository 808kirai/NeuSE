"""
checks the coordinator alerts, raised when an area gets several strong sightings in a short time
texts go through a fake sender here, the real one needs a huawei cloud smn topic
"""
import sys
import threading
from datetime import datetime, timedelta, timezone

import pytest

from command_center import alerts, store
from command_center.app import create_app

here = (-1.2864, 36.8172)


def detection(uid, unit="unit-1", minutes_ago=0, lat=here[0], lon=here[1], confidence=0.9):
    moment = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return {"uid": uid, "unit_id": unit, "timestamp": moment.isoformat(timespec="seconds"), "latitude": lat,
            "longitude": lon, "confidence": confidence, "thermal_peak": 200, "bbox": [1, 2, 3, 4],
            "detector": "person-check", "latency_ms": 20.0, "position_source": "projected"}


def uids(start, count):
    return [f"{i:032x}" for i in range(start, start + count)]


class fake_sender:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    def __call__(self, subject, text):
        if self.fail:
            raise ConnectionError("smn did not answer")
        self.sent.append((subject, text))
        return "message-1"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "center.db"
    store.setup(path)
    return path


defaults = alerts.settings({})


def test_three_strong_sightings_in_one_area_raise_one_alert(db):
    store.save(db, [detection(u, minutes_ago=i) for i, u in enumerate(uids(0, 3))])
    send = fake_sender()
    raised = alerts.check(db, defaults, send)
    assert len(raised) == 1 and len(send.sent) == 1
    alert = store.alerts(db)[0]
    assert alert["sightings"] == 3 and alert["units"] == ["unit-1"] and alert["delivered"] is True
    assert alert["latitude"] == pytest.approx(here[0], abs=1e-5)
    assert "3 strong sightings" in send.sent[0][1]


def test_too_few_weak_or_old_sightings_raise_nothing(db):
    store.save(db, [detection(u) for u in uids(0, 2)])
    store.save(db, [detection(u, confidence=0.5) for u in uids(10, 5)])
    store.save(db, [detection(u, minutes_ago=30) for u in uids(20, 5)])
    assert alerts.check(db, defaults, fake_sender()) == []


def test_an_area_already_alerted_stays_quiet(db):
    store.save(db, [detection(u) for u in uids(0, 3)])
    send = fake_sender()
    alerts.check(db, defaults, send)
    store.save(db, [detection(u) for u in uids(10, 3)])
    assert alerts.check(db, defaults, send) == [] and len(send.sent) == 1


def test_an_area_alerts_again_after_the_quiet_time(db):
    store.save(db, [detection(u) for u in uids(0, 3)])
    alerts.check(db, defaults, None, now=datetime.now(timezone.utc) - timedelta(minutes=40))
    # the first alert was raised 40 minutes ago, past the 30 minute quiet time
    assert len(alerts.check(db, defaults, None)) == 1


def test_two_separate_areas_get_two_alerts(db):
    far = here[0] + 0.01
    store.save(db, [detection(u) for u in uids(0, 3)] + [detection(u, unit="unit-2", lat=far) for u in uids(10, 3)])
    raised = alerts.check(db, defaults, None)
    assert len(raised) == 2 and {a["units"][0] for a in raised} == {"unit-1", "unit-2"}


def test_a_failed_text_still_keeps_the_alert(db):
    store.save(db, [detection(u) for u in uids(0, 3)])
    alerts.check(db, defaults, fake_sender(fail=True))
    alert = store.alerts(db)[0]
    assert alert["delivered"] is False and alert["sent_via"] == "smn" and "smn did not answer" in alert["error"]


def test_texts_are_short_and_have_no_square_brackets():
    area = {"latitude": -1.2, "longitude": 36.8, "sightings": 3, "units": ["unit-[1]"] * 100,
            "highest_confidence": 0.95}
    text = alerts.text_for(area, 10, "https://neuse.example")
    assert len(text) <= alerts.max_text and "[" not in text and "]" not in text


def test_the_map_address_is_added_to_texts():
    area = {"latitude": -1.2, "longitude": 36.8, "sightings": 3, "units": ["unit-1"], "highest_confidence": 0.95}
    assert alerts.text_for(area, 10, "https://neuse.example").endswith("Map: https://neuse.example")


def test_settings_come_from_the_environment():
    config = alerts.settings({"NEUSE_ALERT_MIN_SIGHTINGS": "5", "NEUSE_PUBLIC_URL": "https://x.example/",
                              "NEUSE_SMN_TOPIC_URN": "urn:smn:af-south-1:1:neuse"})
    assert config.min_sightings == 5 and config.public_url == "https://x.example"
    # a topic without keys is not enough to send
    assert not config.texts_on()


texts_env = {"NEUSE_SMN_TOPIC_URN": "urn:smn:af-south-1:1:neuse", "NEUSE_SMN_REGION": "af-south-1",
             "NEUSE_SMN_PROJECT_ID": "p", "CLOUD_SDK_AK": "ak", "CLOUD_SDK_SK": "sk"}


def test_a_missing_sdk_is_reported_and_alerts_still_work(db, monkeypatch):
    monkeypatch.setitem(sys.modules, "huaweicloudsdkcore", None)
    monkeypatch.setitem(sys.modules, "huaweicloudsdkcore.auth.credentials", None)
    watch = alerts.watcher(db, alerts.settings(texts_env))
    assert watch.sender is None and "not installed" in watch.problem
    store.save(db, [detection(u) for u in uids(0, 3)])
    assert len(watch.check_now()) == 1


def test_sightings_arriving_during_a_check_get_checked(db, monkeypatch):
    watch = alerts.watcher(db, defaults)
    real = alerts.check
    calls = []

    def slow_check(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            # a second upload asks for a check while this one is running
            store.save(db, [detection(u) for u in uids(0, 3)])
            assert watch.check_now() == []
        return real(*args, **kwargs)

    monkeypatch.setattr(alerts, "check", slow_check)
    raised = watch.check_now()
    assert len(calls) == 2 and len(raised) == 1


def test_background_checks_also_raise_alerts(db):
    watch = alerts.watcher(db, defaults)
    store.save(db, [detection(u) for u in uids(0, 3)])
    watch.check_soon()
    for _ in range(100):
        if store.alerts(db):
            break
        threading.Event().wait(0.05)
    assert len(store.alerts(db)) == 1


# through the command center

def center(tmp_path, monkeypatch, send=None, env=None):
    if send is not None:
        monkeypatch.setattr(alerts, "smn_sender", lambda config: send)
    return create_app(tmp_path / "center.db", token="", unit_tokens={},
                      alert_settings=alerts.settings(env or {}), background_alerts=False).test_client()


def test_sightings_sent_by_units_raise_an_alert_on_the_map(tmp_path, monkeypatch):
    send = fake_sender()
    client = center(tmp_path, monkeypatch, send, texts_env)
    for batch in (uids(0, 2), uids(2, 1)):
        assert client.post("/api/v1/detections", json={"detections": [detection(u) for u in batch]}).status_code == 200
    data = client.get("/api/v1/alerts").get_json()
    assert len(data["alerts"]) == 1 and data["texts"] is True and data["problem"] is None
    assert len(send.sent) == 1


def test_without_texts_alerts_are_only_on_the_map(tmp_path, monkeypatch):
    client = center(tmp_path, monkeypatch)
    client.post("/api/v1/detections", json={"detections": [detection(u) for u in uids(0, 3)]})
    data = client.get("/api/v1/alerts").get_json()
    assert data["texts"] is False and data["alerts"][0]["sent_via"] is None


def test_resent_sightings_do_not_raise_a_second_alert(tmp_path, monkeypatch):
    send = fake_sender()
    client = center(tmp_path, monkeypatch, send, texts_env)
    body = {"detections": [detection(u) for u in uids(0, 3)]}
    client.post("/api/v1/detections", json=body)
    client.post("/api/v1/detections", json=body)
    assert len(send.sent) == 1


def test_alerts_follow_the_token_rules(tmp_path):
    client = create_app(tmp_path / "c.db", token="boss", unit_tokens={"unit-1": "one"},
                        alert_settings=alerts.settings({})).test_client()
    assert client.get("/api/v1/alerts").status_code == 401
    assert client.get("/api/v1/alerts", headers={"Authorization": "Bearer one"}).status_code == 403
    assert client.get("/api/v1/alerts", headers={"Authorization": "Bearer boss"}).status_code == 200


@pytest.mark.parametrize("hours", ["0", "200", "soon"])
def test_bad_alert_windows_are_refused(tmp_path, monkeypatch, hours):
    assert center(tmp_path, monkeypatch).get(f"/api/v1/alerts?hours={hours}").status_code == 400


def test_the_map_page_has_the_alerts_list(tmp_path, monkeypatch):
    page = center(tmp_path, monkeypatch).get("/").get_data(as_text=True)
    assert 'id="alerts"' in page and "/api/v1/alerts" in page
