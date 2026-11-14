"""
alerts for coordinators when an area gets several strong sightings in a short time
an alert is always kept and shown on the map, and can also go out as a text message through huawei cloud smn
a text reaches any phone, even on a 2g network, which matters where rescuers work

an alert is raised when an area, sightings within 100 m of each other, has at least min_sightings
with a confidence of at least min_confidence that reached the command center within the last window_minutes
counting by arrival means a unit that was out of range still raises an alert when its backlog comes in,
sightings older than max_age_minutes are left out since the people will have moved on
an area near one already alerted stays quiet for quiet_minutes, so phones are not flooded

to send texts, create an smn topic in huawei cloud with sms subscriptions for the coordinators' phones,
install huaweicloudsdksmn, and set
    NEUSE_SMN_TOPIC_URN, the topic's urn
    NEUSE_SMN_REGION, for example af-south-1, the region huawei cloud suggests for africa
    NEUSE_SMN_PROJECT_ID, the project id of that region
    CLOUD_SDK_AK and CLOUD_SDK_SK, an access key pair, the names huawei cloud's own sdk uses
"""
import os
import sys
import threading
from datetime import datetime, timedelta, timezone

from command_center import reports, store

# smn allows at most 490 characters in a text and no square brackets
max_text = 490


class settings:
    # read when used, so a running command center picks up the environment it was started with
    def __init__(self, environ=None):
        env = os.environ if environ is None else environ
        self.min_sightings = int(env.get("NEUSE_ALERT_MIN_SIGHTINGS") or 3)
        self.min_confidence = float(env.get("NEUSE_ALERT_MIN_CONFIDENCE") or 0.8)
        self.window_minutes = float(env.get("NEUSE_ALERT_WINDOW_MINUTES") or 10)
        self.quiet_minutes = float(env.get("NEUSE_ALERT_QUIET_MINUTES") or 30)
        self.max_age_minutes = float(env.get("NEUSE_ALERT_MAX_AGE_MINUTES") or 120)
        # how near in metres an earlier alert must be to keep a new one quiet
        self.quiet_radius = 200.0
        # the address coordinators open, added to texts so they can go straight to the map
        self.public_url = (env.get("NEUSE_PUBLIC_URL") or "").rstrip("/")
        self.topic_urn = env.get("NEUSE_SMN_TOPIC_URN") or ""
        self.region = env.get("NEUSE_SMN_REGION") or ""
        self.project_id = env.get("NEUSE_SMN_PROJECT_ID") or ""
        self.access_key = env.get("CLOUD_SDK_AK") or ""
        self.secret_key = env.get("CLOUD_SDK_SK") or ""

    def texts_on(self):
        return all((self.topic_urn, self.region, self.project_id, self.access_key, self.secret_key))


def smn_sender(config):
    # returns a function that sends one text through huawei cloud smn, following huawei's own sdk sample
    # raises ImportError when the sdk is not installed
    from huaweicloudsdkcore.auth.credentials import BasicCredentials
    from huaweicloudsdksmn.v2 import PublishMessageRequest, PublishMessageRequestBody, SmnClient
    from huaweicloudsdksmn.v2.region.smn_region import SmnRegion

    client = (SmnClient.new_builder()
              .with_credentials(BasicCredentials(config.access_key, config.secret_key, config.project_id))
              .with_region(SmnRegion.value_of(config.region))
              .build())

    def send(subject, text):
        request = PublishMessageRequest()
        request.topic_urn = config.topic_urn
        request.body = PublishMessageRequestBody(subject=subject, message=text, time_to_live="3600")
        return client.publish_message(request).message_id

    return send


def seen_when(newest, oldest):
    # minutes ago the sightings were made, as "seen just now", "seen 4 min ago" or "seen 12 to 40 min ago"
    if oldest <= 1:
        return "seen just now"
    if newest == oldest:
        return f"seen {newest} min ago"
    return f"seen {newest} to {oldest} min ago"


def text_for(area, public_url):
    # the alert in plain words, short enough for one text and free of square brackets
    confidence = f", highest confidence {area['highest_confidence']}" if area["highest_confidence"] is not None else ""
    text = (f"NeuSE alert: {reports.count(area['sightings'], 'strong sighting')} near "
            f"{area['latitude']}, {area['longitude']} by {', '.join(area['units'])}, "
            f"{seen_when(area['newest_minutes'], area['oldest_minutes'])}{confidence}.")
    if public_url:
        text += f" Map: {public_url}"
    return text.replace("[", "(").replace("]", ")")[:max_text]


def check(db, config=None, sender=None, now=None):
    # looks for areas that need an alert, saves each one and sends it, returns the alerts raised
    # sender is a function taking a subject and text, None means alerts are only kept for the map
    config = config or settings()
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    arrived = (now - timedelta(minutes=config.window_minutes)).isoformat(timespec="seconds")
    seen_after = (now - timedelta(minutes=config.max_age_minutes)).isoformat(timespec="seconds")
    strong = [s for s in store.sightings_arrived(db, arrived, seen_after)
              if s["confidence"] is not None and s["confidence"] >= config.min_confidence]

    quiet_since = (now - timedelta(minutes=config.quiet_minutes)).isoformat(timespec="seconds")
    recent = store.alerts(db, since=quiet_since)
    raised = []
    for group in reports.group_areas(strong):
        points = group["points"]
        if len(points) < config.min_sightings:
            continue
        centre = group["centre"]
        if any(reports.distance(centre, (a["latitude"], a["longitude"])) <= config.quiet_radius for a in recent + raised):
            continue
        ages = [max(0, round((now - datetime.fromisoformat(p["timestamp_utc"])).total_seconds() / 60)) for p in points]
        area = {
            "newest_minutes": min(ages),
            "oldest_minutes": max(ages),
            "latitude": round(centre[0], 5),
            "longitude": round(centre[1], 5),
            "sightings": len(points),
            "units": sorted({p["unit_id"] for p in points}),
            "highest_confidence": round(max(p["confidence"] for p in points), 2),
        }
        alert = {**area, "created_at": now.isoformat(timespec="seconds"),
                 "text": text_for(area, config.public_url),
                 "sent_via": None, "delivered": 0, "error": None}
        if sender is not None:
            try:
                sender("NeuSE alert", alert["text"])
                alert["sent_via"], alert["delivered"] = "smn", 1
            except Exception as err:  # the alert must be kept whatever the text service does
                alert["sent_via"], alert["error"] = "smn", f"{type(err).__name__}: {err}"[:500]
        store.save_alert(db, alert)
        raised.append(alert)
    return raised


class watcher:
    # checks for alerts after sightings arrive, one check at a time and away from the unit's request
    def __init__(self, db, config=None):
        self.db = db
        self.config = config or settings()
        self.lock = threading.Lock()
        # set when sightings arrive, so a check already running looks again instead of missing them
        self.wanted = False
        self.sender = None
        self.problem = None
        if self.config.texts_on():
            try:
                self.sender = smn_sender(self.config)
            except ImportError:
                self.problem = "texts are set up but huaweicloudsdksmn is not installed"
            except Exception as err:  # a wrong key or region is reported, alerts still show on the map
                self.problem = f"texts are set up but could not start, {type(err).__name__}: {err}"

    def check_now(self):
        # one check at a time, a check asked for while another runs makes that one look again when it finishes
        # wanted is looked at again after the lock is let go, so a request arriving right then is not lost
        self.wanted = True
        raised = []
        while self.wanted:
            if not self.lock.acquire(blocking=False):
                return raised
            try:
                while self.wanted:
                    self.wanted = False
                    raised += check(self.db, self.config, self.sender)
            finally:
                self.lock.release()
        return raised

    def check_soon(self):
        # runs a check in the background so a unit's sync is never held up by a slow text service
        threading.Thread(target=self.quietly, daemon=True).start()

    def quietly(self):
        try:
            self.check_now()
        except Exception as err:  # a database hiccup must not kill the thread with a traceback on every sync
            print(f"warning, alert check failed, {type(err).__name__}: {err}", file=sys.stderr)
