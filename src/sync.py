"""
store and forward sync from a field unit to the command center
detections wait in the local database until the command center confirms each one
when the link is down nothing is lost, the unit keeps detecting and sends the backlog once it is back

newest detections go first, so coordinators see the current picture as soon as a link returns
resending is safe, the command center ignores detection ids it already has

usage
    python -m src.sync --url http://203.0.113.10:8000
    python -m src.sync --once
"""
import argparse
import gzip
import json
import logging
import random
import sqlite3
import time
import urllib.error
import urllib.request
from contextlib import closing

from src.config import config
from src.logger import connect, data_logger, now, read_status

log = logging.getLogger(__name__)

# fields sent for each detection
columns = ("uid", "unit_id", "timestamp", "latitude", "longitude", "confidence",
           "thermal_peak", "bbox", "detector", "latency_ms")


class send_error(Exception):
    # the batch did not reach the command center, so it stays queued
    pass


def pending(limit):
    # the newest detections the command center has not confirmed yet
    with closing(connect()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"""
            select {", ".join(columns)} from detections
            where synced_at is null
            order by id desc limit ?
        """, (limit,)).fetchall()

    detections = []
    for row in rows:
        detection = dict(row)
        # boxes are saved as text like [x, y, w, h]
        try:
            detection["bbox"] = json.loads(detection["bbox"]) if detection["bbox"] else None
        except ValueError:
            detection["bbox"] = None
        detections.append(detection)
    return detections


def pending_count():
    with closing(connect()) as conn:
        return conn.execute("select count(*) from detections where synced_at is null").fetchone()[0]


def mark(accepted, rejected):
    # accepted detections are done
    # rejected ones are kept locally with the reason and not sent again, so one bad row cannot block the queue
    when = now()
    with closing(connect()) as conn, conn:
        conn.executemany("update detections set synced_at = ? where uid = ? and synced_at is null",
                         [(when, uid) for uid in accepted])
        conn.executemany("update detections set synced_at = ?, sync_error = ? where uid = ? and synced_at is null",
                         [(when, reason, uid) for uid, reason in rejected.items()])


def poster(url, path, token=None, timeout=15):
    # returns a function that posts json to one command center address and gives back its reply
    endpoint = url.rstrip("/") + path
    headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def post(payload):
        # compressed, since field links can be slow and paid by the megabyte
        body = gzip.compress(json.dumps(payload).encode("utf-8"))
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            raise send_error(f"command center answered {err.code}") from err
        except (urllib.error.URLError, OSError) as err:
            raise send_error(f"could not reach the command center, {err}") from err
        except ValueError as err:
            raise send_error("command center reply was not json") from err

    return post


def http_sender(url, token=None, timeout=15):
    # returns a function that posts one batch of detections
    post = poster(url, "/api/v1/detections", token, timeout)
    return lambda detections: post({"detections": detections})


def http_status_sender(url, token=None, timeout=15):
    # returns a function that posts the unit's status note
    return poster(url, "/api/v1/status", token, timeout)


def status_note():
    # what the command center hears from a unit even when it has nothing new
    # where the unit is, what it runs and how many detections are still waiting
    status = read_status() or {}
    return {
        "unit_id": config.unit_id,
        "queued": pending_count(),
        "latitude": status.get("latitude", config.default_lat),
        "longitude": status.get("longitude", config.default_lon),
        "detector": status.get("detector"),
        "frames": status.get("frames"),
        "updated_at": status.get("updated_at") or now(),
    }


def sync_once(send, batch=None):
    # sends queued detections a batch at a time until the queue is empty or the link fails
    batch = batch or config.sync_batch
    synced = 0
    while True:
        rows = pending(batch)
        if not rows:
            return {"synced": synced, "pending": 0, "error": None}

        try:
            reply = send(rows)
        except send_error as err:
            return {"synced": synced, "pending": pending_count(), "error": str(err)}
        if not isinstance(reply, dict):
            return {"synced": synced, "pending": pending_count(), "error": "command center reply was not understood"}

        # only trust ids that were in this batch
        sent = {row["uid"] for row in rows}
        accepted = reply.get("accepted")
        accepted = [uid for uid in accepted if isinstance(uid, str) and uid in sent] if isinstance(accepted, list) else []
        rejected = reply.get("rejected")
        rejected = {uid: str(why) for uid, why in rejected.items() if uid in sent} if isinstance(rejected, dict) else {}
        mark(accepted, rejected)
        synced += len(accepted)

        # nothing taken means the same batch would come straight back, so stop and wait instead
        if not accepted and not rejected:
            return {"synced": synced, "pending": pending_count(), "error": "command center took nothing from the batch"}


def run(send, interval=None, max_wait=None, stop=lambda: False, sleep=time.sleep, jitter=random.random,
        send_status=None):
    # keeps syncing until stop() is true
    # send_status, when given, also posts the unit's status note each round
    # after a failure the wait doubles up to max_wait, and goes back to interval once a sync works
    # the random part spreads retries so many units do not all hit the command center at once
    interval = interval or config.sync_interval
    max_wait = max_wait or config.sync_max_wait
    failures = 0
    while not stop():
        result = sync_once(send)
        if send_status is not None:
            try:
                send_status(status_note())
            except send_error:
                # the detections are what matter, a missed status note is simply sent next round
                pass
        if result["error"]:
            failures += 1
            wait = min(max_wait, interval * 2 ** failures) * (0.5 + jitter() / 2)
            log.warning("sync failed, %s queued, trying again in %.0f s, %s", result["pending"], wait, result["error"])
        else:
            failures = 0
            wait = interval
            if result["synced"]:
                log.info("sent %s detections", result["synced"])
        sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="send logged detections to the command center")
    parser.add_argument("--url", default=config.sync_url, help="command center address, or set NEUSE_SYNC_URL")
    parser.add_argument("--token", default=config.sync_token, help="shared secret, or set NEUSE_SYNC_TOKEN")
    parser.add_argument("--once", action="store_true", help="send what is queued and stop")
    parser.add_argument("--interval", type=float, default=config.sync_interval, help="seconds between checks")
    args = parser.parse_args()
    if not args.url:
        parser.error("no command center address, use --url or set NEUSE_SYNC_URL")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # make sure the table exists before the first read
    data_logger()
    send = http_sender(args.url, args.token)
    send_status = http_status_sender(args.url, args.token)

    if args.once:
        result = sync_once(send)
        try:
            send_status(status_note())
        except send_error:
            pass
        print(f"sent {result['synced']}, still queued {result['pending']}" +
              (f", {result['error']}" if result["error"] else ""))
        return
    log.info("unit %s syncing to %s", config.unit_id, args.url)
    run(send, interval=args.interval, send_status=send_status)


if __name__ == "__main__":
    main()
